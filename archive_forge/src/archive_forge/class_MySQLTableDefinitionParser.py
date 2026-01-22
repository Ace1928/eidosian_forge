import re
from .enumerated import ENUM
from .enumerated import SET
from .types import DATETIME
from .types import TIME
from .types import TIMESTAMP
from ... import log
from ... import types as sqltypes
from ... import util
@log.class_logger
class MySQLTableDefinitionParser:
    """Parses the results of a SHOW CREATE TABLE statement."""

    def __init__(self, dialect, preparer):
        self.dialect = dialect
        self.preparer = preparer
        self._prep_regexes()

    def parse(self, show_create, charset):
        state = ReflectedState()
        state.charset = charset
        for line in re.split('\\r?\\n', show_create):
            if line.startswith('  ' + self.preparer.initial_quote):
                self._parse_column(line, state)
            elif line.startswith(') '):
                self._parse_table_options(line, state)
            elif line == ')':
                pass
            elif line.startswith('CREATE '):
                self._parse_table_name(line, state)
            elif 'PARTITION' in line:
                self._parse_partition_options(line, state)
            elif not line:
                pass
            else:
                type_, spec = self._parse_constraints(line)
                if type_ is None:
                    util.warn('Unknown schema content: %r' % line)
                elif type_ == 'key':
                    state.keys.append(spec)
                elif type_ == 'fk_constraint':
                    state.fk_constraints.append(spec)
                elif type_ == 'ck_constraint':
                    state.ck_constraints.append(spec)
                else:
                    pass
        return state

    def _check_view(self, sql: str) -> bool:
        return bool(self._re_is_view.match(sql))

    def _parse_constraints(self, line):
        """Parse a KEY or CONSTRAINT line.

        :param line: A line of SHOW CREATE TABLE output
        """
        m = self._re_key.match(line)
        if m:
            spec = m.groupdict()
            spec['columns'] = self._parse_keyexprs(spec['columns'])
            if spec['version_sql']:
                m2 = self._re_key_version_sql.match(spec['version_sql'])
                if m2 and m2.groupdict()['parser']:
                    spec['parser'] = m2.groupdict()['parser']
            if spec['parser']:
                spec['parser'] = self.preparer.unformat_identifiers(spec['parser'])[0]
            return ('key', spec)
        m = self._re_fk_constraint.match(line)
        if m:
            spec = m.groupdict()
            spec['table'] = self.preparer.unformat_identifiers(spec['table'])
            spec['local'] = [c[0] for c in self._parse_keyexprs(spec['local'])]
            spec['foreign'] = [c[0] for c in self._parse_keyexprs(spec['foreign'])]
            return ('fk_constraint', spec)
        m = self._re_ck_constraint.match(line)
        if m:
            spec = m.groupdict()
            return ('ck_constraint', spec)
        m = self._re_partition.match(line)
        if m:
            return ('partition', line)
        return (None, line)

    def _parse_table_name(self, line, state):
        """Extract the table name.

        :param line: The first line of SHOW CREATE TABLE
        """
        regex, cleanup = self._pr_name
        m = regex.match(line)
        if m:
            state.table_name = cleanup(m.group('name'))

    def _parse_table_options(self, line, state):
        """Build a dictionary of all reflected table-level options.

        :param line: The final line of SHOW CREATE TABLE output.
        """
        options = {}
        if line and line != ')':
            rest_of_line = line
            for regex, cleanup in self._pr_options:
                m = regex.search(rest_of_line)
                if not m:
                    continue
                directive, value = (m.group('directive'), m.group('val'))
                if cleanup:
                    value = cleanup(value)
                options[directive.lower()] = value
                rest_of_line = regex.sub('', rest_of_line)
        for nope in ('auto_increment', 'data directory', 'index directory'):
            options.pop(nope, None)
        for opt, val in options.items():
            state.table_options['%s_%s' % (self.dialect.name, opt)] = val

    def _parse_partition_options(self, line, state):
        options = {}
        new_line = line[:]
        while new_line.startswith('(') or new_line.startswith(' '):
            new_line = new_line[1:]
        for regex, cleanup in self._pr_options:
            m = regex.search(new_line)
            if not m or 'PARTITION' not in regex.pattern:
                continue
            directive = m.group('directive')
            directive = directive.lower()
            is_subpartition = directive == 'subpartition'
            if directive == 'partition' or is_subpartition:
                new_line = new_line.replace(') */', '')
                new_line = new_line.replace(',', '')
                if is_subpartition and new_line.endswith(')'):
                    new_line = new_line[:-1]
                if self.dialect.name == 'mariadb' and new_line.endswith(')'):
                    if 'MAXVALUE' in new_line or 'MINVALUE' in new_line or 'ENGINE' in new_line:
                        new_line = new_line[:-1]
                defs = '%s_%s_definitions' % (self.dialect.name, directive)
                options[defs] = new_line
            else:
                directive = directive.replace(' ', '_')
                value = m.group('val')
                if cleanup:
                    value = cleanup(value)
                options[directive] = value
            break
        for opt, val in options.items():
            part_def = '%s_partition_definitions' % self.dialect.name
            subpart_def = '%s_subpartition_definitions' % self.dialect.name
            if opt == part_def or opt == subpart_def:
                if opt not in state.table_options:
                    state.table_options[opt] = val
                else:
                    state.table_options[opt] = '%s, %s' % (state.table_options[opt], val)
            else:
                state.table_options['%s_%s' % (self.dialect.name, opt)] = val

    def _parse_column(self, line, state):
        """Extract column details.

        Falls back to a 'minimal support' variant if full parse fails.

        :param line: Any column-bearing line from SHOW CREATE TABLE
        """
        spec = None
        m = self._re_column.match(line)
        if m:
            spec = m.groupdict()
            spec['full'] = True
        else:
            m = self._re_column_loose.match(line)
            if m:
                spec = m.groupdict()
                spec['full'] = False
        if not spec:
            util.warn('Unknown column definition %r' % line)
            return
        if not spec['full']:
            util.warn('Incomplete reflection of column definition %r' % line)
        name, type_, args = (spec['name'], spec['coltype'], spec['arg'])
        try:
            col_type = self.dialect.ischema_names[type_]
        except KeyError:
            util.warn("Did not recognize type '%s' of column '%s'" % (type_, name))
            col_type = sqltypes.NullType
        if args is None or args == '':
            type_args = []
        elif args[0] == "'" and args[-1] == "'":
            type_args = self._re_csv_str.findall(args)
        else:
            type_args = [int(v) for v in self._re_csv_int.findall(args)]
        type_kw = {}
        if issubclass(col_type, (DATETIME, TIME, TIMESTAMP)):
            if type_args:
                type_kw['fsp'] = type_args.pop(0)
        for kw in ('unsigned', 'zerofill'):
            if spec.get(kw, False):
                type_kw[kw] = True
        for kw in ('charset', 'collate'):
            if spec.get(kw, False):
                type_kw[kw] = spec[kw]
        if issubclass(col_type, (ENUM, SET)):
            type_args = _strip_values(type_args)
            if issubclass(col_type, SET) and '' in type_args:
                type_kw['retrieve_as_bitwise'] = True
        type_instance = col_type(*type_args, **type_kw)
        col_kw = {}
        col_kw['nullable'] = True
        if spec.get('notnull', False) == 'NOT NULL':
            col_kw['nullable'] = False
        if spec.get('notnull_generated', False) == 'NOT NULL':
            col_kw['nullable'] = False
        if spec.get('autoincr', False):
            col_kw['autoincrement'] = True
        elif issubclass(col_type, sqltypes.Integer):
            col_kw['autoincrement'] = False
        default = spec.get('default', None)
        if default == 'NULL':
            default = None
        comment = spec.get('comment', None)
        if comment is not None:
            comment = cleanup_text(comment)
        sqltext = spec.get('generated')
        if sqltext is not None:
            computed = dict(sqltext=sqltext)
            persisted = spec.get('persistence')
            if persisted is not None:
                computed['persisted'] = persisted == 'STORED'
            col_kw['computed'] = computed
        col_d = dict(name=name, type=type_instance, default=default, comment=comment)
        col_d.update(col_kw)
        state.columns.append(col_d)

    def _describe_to_create(self, table_name, columns):
        """Re-format DESCRIBE output as a SHOW CREATE TABLE string.

        DESCRIBE is a much simpler reflection and is sufficient for
        reflecting views for runtime use.  This method formats DDL
        for columns only- keys are omitted.

        :param columns: A sequence of DESCRIBE or SHOW COLUMNS 6-tuples.
          SHOW FULL COLUMNS FROM rows must be rearranged for use with
          this function.
        """
        buffer = []
        for row in columns:
            name, col_type, nullable, default, extra = (row[i] for i in (0, 1, 2, 4, 5))
            line = [' ']
            line.append(self.preparer.quote_identifier(name))
            line.append(col_type)
            if not nullable:
                line.append('NOT NULL')
            if default:
                if 'auto_increment' in default:
                    pass
                elif col_type.startswith('timestamp') and default.startswith('C'):
                    line.append('DEFAULT')
                    line.append(default)
                elif default == 'NULL':
                    line.append('DEFAULT')
                    line.append(default)
                else:
                    line.append('DEFAULT')
                    line.append("'%s'" % default.replace("'", "''"))
            if extra:
                line.append(extra)
            buffer.append(' '.join(line))
        return ''.join(['CREATE TABLE %s (\n' % self.preparer.quote_identifier(table_name), ',\n'.join(buffer), '\n) '])

    def _parse_keyexprs(self, identifiers):
        """Unpack '"col"(2),"col" ASC'-ish strings into components."""
        return [(colname, int(length) if length else None, modifiers) for colname, length, modifiers in self._re_keyexprs.findall(identifiers)]

    def _prep_regexes(self):
        """Pre-compile regular expressions."""
        self._re_columns = []
        self._pr_options = []
        _final = self.preparer.final_quote
        quotes = dict(zip(('iq', 'fq', 'esc_fq'), [re.escape(s) for s in (self.preparer.initial_quote, _final, self.preparer._escape_identifier(_final))]))
        self._pr_name = _pr_compile('^CREATE (?:\\w+ +)?TABLE +%(iq)s(?P<name>(?:%(esc_fq)s|[^%(fq)s])+)%(fq)s +\\($' % quotes, self.preparer._unescape_identifier)
        self._re_is_view = _re_compile('^CREATE(?! TABLE)(\\s.*)?\\sVIEW')
        self._re_keyexprs = _re_compile('(?:(?:%(iq)s((?:%(esc_fq)s|[^%(fq)s])+)%(fq)s)(?:\\((\\d+)\\))?(?: +(ASC|DESC))?(?=\\,|$))+' % quotes)
        self._re_csv_str = _re_compile('\\x27(?:\\x27\\x27|[^\\x27])*\\x27')
        self._re_csv_int = _re_compile('\\d+')
        self._re_column = _re_compile("  %(iq)s(?P<name>(?:%(esc_fq)s|[^%(fq)s])+)%(fq)s +(?P<coltype>\\w+)(?:\\((?P<arg>(?:\\d+|\\d+,\\d+|(?:'(?:''|[^'])*',?)+))\\))?(?: +(?P<unsigned>UNSIGNED))?(?: +(?P<zerofill>ZEROFILL))?(?: +CHARACTER SET +(?P<charset>[\\w_]+))?(?: +COLLATE +(?P<collate>[\\w_]+))?(?: +(?P<notnull>(?:NOT )?NULL))?(?: +DEFAULT +(?P<default>(?:NULL|'(?:''|[^'])*'|[\\-\\w\\.\\(\\)]+(?: +ON UPDATE [\\-\\w\\.\\(\\)]+)?)))?(?: +(?:GENERATED ALWAYS)? ?AS +(?P<generated>\\(.*\\))? ?(?P<persistence>VIRTUAL|STORED)?(?: +(?P<notnull_generated>(?:NOT )?NULL))?)?(?: +(?P<autoincr>AUTO_INCREMENT))?(?: +COMMENT +'(?P<comment>(?:''|[^'])*)')?(?: +COLUMN_FORMAT +(?P<colfmt>\\w+))?(?: +STORAGE +(?P<storage>\\w+))?(?: +(?P<extra>.*))?,?$" % quotes)
        self._re_column_loose = _re_compile('  %(iq)s(?P<name>(?:%(esc_fq)s|[^%(fq)s])+)%(fq)s +(?P<coltype>\\w+)(?:\\((?P<arg>(?:\\d+|\\d+,\\d+|\\x27(?:\\x27\\x27|[^\\x27])+\\x27))\\))?.*?(?P<notnull>(?:NOT )NULL)?' % quotes)
        self._re_key = _re_compile('  (?:(?P<type>\\S+) )?KEY(?: +%(iq)s(?P<name>(?:%(esc_fq)s|[^%(fq)s])+)%(fq)s)?(?: +USING +(?P<using_pre>\\S+))? +\\((?P<columns>.+?)\\)(?: +USING +(?P<using_post>\\S+))?(?: +KEY_BLOCK_SIZE *[ =]? *(?P<keyblock>\\S+))?(?: +WITH PARSER +(?P<parser>\\S+))?(?: +COMMENT +(?P<comment>(\\x27\\x27|\\x27([^\\x27])*?\\x27)+))?(?: +/\\*(?P<version_sql>.+)\\*/ *)?,?$' % quotes)
        self._re_key_version_sql = _re_compile('\\!\\d+ (?: *WITH PARSER +(?P<parser>\\S+) *)?')
        kw = quotes.copy()
        kw['on'] = 'RESTRICT|CASCADE|SET NULL|NO ACTION'
        self._re_fk_constraint = _re_compile('  CONSTRAINT +%(iq)s(?P<name>(?:%(esc_fq)s|[^%(fq)s])+)%(fq)s +FOREIGN KEY +\\((?P<local>[^\\)]+?)\\) REFERENCES +(?P<table>%(iq)s[^%(fq)s]+%(fq)s(?:\\.%(iq)s[^%(fq)s]+%(fq)s)?) +\\((?P<foreign>(?:%(iq)s[^%(fq)s]+%(fq)s(?: *, *)?)+)\\)(?: +(?P<match>MATCH \\w+))?(?: +ON DELETE (?P<ondelete>%(on)s))?(?: +ON UPDATE (?P<onupdate>%(on)s))?' % kw)
        self._re_ck_constraint = _re_compile('  CONSTRAINT +%(iq)s(?P<name>(?:%(esc_fq)s|[^%(fq)s])+)%(fq)s +CHECK +\\((?P<sqltext>.+)\\),?' % kw)
        self._re_partition = _re_compile('(?:.*)(?:SUB)?PARTITION(?:.*)')
        for option in _options_of_type_string:
            self._add_option_string(option)
        for option in ('ENGINE', 'TYPE', 'AUTO_INCREMENT', 'AVG_ROW_LENGTH', 'CHARACTER SET', 'DEFAULT CHARSET', 'CHECKSUM', 'COLLATE', 'DELAY_KEY_WRITE', 'INSERT_METHOD', 'MAX_ROWS', 'MIN_ROWS', 'PACK_KEYS', 'ROW_FORMAT', 'KEY_BLOCK_SIZE', 'STATS_SAMPLE_PAGES'):
            self._add_option_word(option)
        for option in ('PARTITION BY', 'SUBPARTITION BY', 'PARTITIONS', 'SUBPARTITIONS', 'PARTITION', 'SUBPARTITION'):
            self._add_partition_option_word(option)
        self._add_option_regex('UNION', '\\([^\\)]+\\)')
        self._add_option_regex('TABLESPACE', '.*? STORAGE DISK')
        self._add_option_regex('RAID_TYPE', '\\w+\\s+RAID_CHUNKS\\s*\\=\\s*\\w+RAID_CHUNKSIZE\\s*=\\s*\\w+')
    _optional_equals = '(?:\\s*(?:=\\s*)|\\s+)'

    def _add_option_string(self, directive):
        regex = "(?P<directive>%s)%s'(?P<val>(?:[^']|'')*?)'(?!')" % (re.escape(directive), self._optional_equals)
        self._pr_options.append(_pr_compile(regex, cleanup_text))

    def _add_option_word(self, directive):
        regex = '(?P<directive>%s)%s(?P<val>\\w+)' % (re.escape(directive), self._optional_equals)
        self._pr_options.append(_pr_compile(regex))

    def _add_partition_option_word(self, directive):
        if directive == 'PARTITION BY' or directive == 'SUBPARTITION BY':
            regex = '(?<!\\S)(?P<directive>%s)%s(?P<val>\\w+.*)' % (re.escape(directive), self._optional_equals)
        elif directive == 'SUBPARTITIONS' or directive == 'PARTITIONS':
            regex = '(?<!\\S)(?P<directive>%s)%s(?P<val>\\d+)' % (re.escape(directive), self._optional_equals)
        else:
            regex = '(?<!\\S)(?P<directive>%s)(?!\\S)' % (re.escape(directive),)
        self._pr_options.append(_pr_compile(regex))

    def _add_option_regex(self, directive, regex):
        regex = '(?P<directive>%s)%s(?P<val>%s)' % (re.escape(directive), self._optional_equals, regex)
        self._pr_options.append(_pr_compile(regex))