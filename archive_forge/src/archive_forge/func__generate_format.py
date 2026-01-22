import decimal
import re
from .exceptions import JsonSchemaDefinitionException
from .generator import CodeGenerator, enforce_list
def _generate_format(self, format_name, regexp_name, regexp):
    if self._definition['format'] == format_name:
        if not regexp_name in self._compile_regexps:
            self._compile_regexps[regexp_name] = re.compile(regexp)
        with self.l('if not REGEX_PATTERNS["{}"].match({variable}):', regexp_name):
            self.exc('{name} must be {}', format_name, rule='format')