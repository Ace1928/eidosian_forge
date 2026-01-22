from __future__ import print_function, absolute_import, division
import datetime
import base64
import binascii
import re
import sys
import types
import warnings
from ruamel.yaml.error import (MarkedYAMLError, MarkedYAMLFutureWarning,
from ruamel.yaml.nodes import *                               # NOQA
from ruamel.yaml.nodes import (SequenceNode, MappingNode, ScalarNode)
from ruamel.yaml.compat import (utf8, builtins_module, to_str, PY2, PY3,  # NOQA
from ruamel.yaml.comments import *                               # NOQA
from ruamel.yaml.comments import (CommentedMap, CommentedOrderedMap, CommentedSet,
from ruamel.yaml.scalarstring import (SingleQuotedScalarString, DoubleQuotedScalarString,
from ruamel.yaml.scalarint import ScalarInt, BinaryInt, OctalInt, HexInt, HexCapsInt
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarbool import ScalarBoolean
from ruamel.yaml.timestamp import TimeStamp
from ruamel.yaml.util import RegExp
class RoundTripConstructor(SafeConstructor):
    """need to store the comments on the node itself,
    as well as on the items
    """

    def construct_scalar(self, node):
        if not isinstance(node, ScalarNode):
            raise ConstructorError(None, None, 'expected a scalar node, but found %s' % node.id, node.start_mark)
        if node.style == '|' and isinstance(node.value, text_type):
            lss = LiteralScalarString(node.value, anchor=node.anchor)
            if node.comment and node.comment[1]:
                lss.comment = node.comment[1][0]
            return lss
        if node.style == '>' and isinstance(node.value, text_type):
            fold_positions = []
            idx = -1
            while True:
                idx = node.value.find('\x07', idx + 1)
                if idx < 0:
                    break
                fold_positions.append(idx - len(fold_positions))
            fss = FoldedScalarString(node.value.replace('\x07', ''), anchor=node.anchor)
            if node.comment and node.comment[1]:
                fss.comment = node.comment[1][0]
            if fold_positions:
                fss.fold_pos = fold_positions
            return fss
        elif bool(self._preserve_quotes) and isinstance(node.value, text_type):
            if node.style == "'":
                return SingleQuotedScalarString(node.value, anchor=node.anchor)
            if node.style == '"':
                return DoubleQuotedScalarString(node.value, anchor=node.anchor)
        if node.anchor:
            return PlainScalarString(node.value, anchor=node.anchor)
        return node.value

    def construct_yaml_int(self, node):
        width = None
        value_su = to_str(self.construct_scalar(node))
        try:
            sx = value_su.rstrip('_')
            underscore = [len(sx) - sx.rindex('_') - 1, False, False]
        except ValueError:
            underscore = None
        except IndexError:
            underscore = None
        value_s = value_su.replace('_', '')
        sign = +1
        if value_s[0] == '-':
            sign = -1
        if value_s[0] in '+-':
            value_s = value_s[1:]
        if value_s == '0':
            return 0
        elif value_s.startswith('0b'):
            if self.resolver.processing_version > (1, 1) and value_s[2] == '0':
                width = len(value_s[2:])
            if underscore is not None:
                underscore[1] = value_su[2] == '_'
                underscore[2] = len(value_su[2:]) > 1 and value_su[-1] == '_'
            return BinaryInt(sign * int(value_s[2:], 2), width=width, underscore=underscore, anchor=node.anchor)
        elif value_s.startswith('0x'):
            if self.resolver.processing_version > (1, 1) and value_s[2] == '0':
                width = len(value_s[2:])
            hex_fun = HexInt
            for ch in value_s[2:]:
                if ch in 'ABCDEF':
                    hex_fun = HexCapsInt
                    break
                if ch in 'abcdef':
                    break
            if underscore is not None:
                underscore[1] = value_su[2] == '_'
                underscore[2] = len(value_su[2:]) > 1 and value_su[-1] == '_'
            return hex_fun(sign * int(value_s[2:], 16), width=width, underscore=underscore, anchor=node.anchor)
        elif value_s.startswith('0o'):
            if self.resolver.processing_version > (1, 1) and value_s[2] == '0':
                width = len(value_s[2:])
            if underscore is not None:
                underscore[1] = value_su[2] == '_'
                underscore[2] = len(value_su[2:]) > 1 and value_su[-1] == '_'
            return OctalInt(sign * int(value_s[2:], 8), width=width, underscore=underscore, anchor=node.anchor)
        elif self.resolver.processing_version != (1, 2) and value_s[0] == '0':
            return sign * int(value_s, 8)
        elif self.resolver.processing_version != (1, 2) and ':' in value_s:
            digits = [int(part) for part in value_s.split(':')]
            digits.reverse()
            base = 1
            value = 0
            for digit in digits:
                value += digit * base
                base *= 60
            return sign * value
        elif self.resolver.processing_version > (1, 1) and value_s[0] == '0':
            if underscore is not None:
                underscore[2] = len(value_su) > 1 and value_su[-1] == '_'
            return ScalarInt(sign * int(value_s), width=len(value_s), underscore=underscore)
        elif underscore:
            underscore[2] = len(value_su) > 1 and value_su[-1] == '_'
            return ScalarInt(sign * int(value_s), width=None, underscore=underscore, anchor=node.anchor)
        elif node.anchor:
            return ScalarInt(sign * int(value_s), width=None, anchor=node.anchor)
        else:
            return sign * int(value_s)

    def construct_yaml_float(self, node):

        def leading_zeros(v):
            lead0 = 0
            idx = 0
            while idx < len(v) and v[idx] in '0.':
                if v[idx] == '0':
                    lead0 += 1
                idx += 1
            return lead0
        m_sign = False
        value_so = to_str(self.construct_scalar(node))
        value_s = value_so.replace('_', '').lower()
        sign = +1
        if value_s[0] == '-':
            sign = -1
        if value_s[0] in '+-':
            m_sign = value_s[0]
            value_s = value_s[1:]
        if value_s == '.inf':
            return sign * self.inf_value
        if value_s == '.nan':
            return self.nan_value
        if self.resolver.processing_version != (1, 2) and ':' in value_s:
            digits = [float(part) for part in value_s.split(':')]
            digits.reverse()
            base = 1
            value = 0.0
            for digit in digits:
                value += digit * base
                base *= 60
            return sign * value
        if 'e' in value_s:
            try:
                mantissa, exponent = value_so.split('e')
                exp = 'e'
            except ValueError:
                mantissa, exponent = value_so.split('E')
                exp = 'E'
            if self.resolver.processing_version != (1, 2):
                if '.' not in mantissa:
                    warnings.warn(MantissaNoDotYAML1_1Warning(node, value_so))
            lead0 = leading_zeros(mantissa)
            width = len(mantissa)
            prec = mantissa.find('.')
            if m_sign:
                width -= 1
            e_width = len(exponent)
            e_sign = exponent[0] in '+-'
            return ScalarFloat(sign * float(value_s), width=width, prec=prec, m_sign=m_sign, m_lead0=lead0, exp=exp, e_width=e_width, e_sign=e_sign, anchor=node.anchor)
        width = len(value_so)
        prec = value_so.index('.')
        lead0 = leading_zeros(value_so)
        return ScalarFloat(sign * float(value_s), width=width, prec=prec, m_sign=m_sign, m_lead0=lead0, anchor=node.anchor)

    def construct_yaml_str(self, node):
        value = self.construct_scalar(node)
        if isinstance(value, ScalarString):
            return value
        if PY3:
            return value
        try:
            return value.encode('ascii')
        except AttributeError:
            return value
        except UnicodeEncodeError:
            return value

    def construct_rt_sequence(self, node, seqtyp, deep=False):
        if not isinstance(node, SequenceNode):
            raise ConstructorError(None, None, 'expected a sequence node, but found %s' % node.id, node.start_mark)
        ret_val = []
        if node.comment:
            seqtyp._yaml_add_comment(node.comment[:2])
            if len(node.comment) > 2:
                seqtyp.yaml_end_comment_extend(node.comment[2], clear=True)
        if node.anchor:
            from ruamel.yaml.serializer import templated_id
            if not templated_id(node.anchor):
                seqtyp.yaml_set_anchor(node.anchor)
        for idx, child in enumerate(node.value):
            ret_val.append(self.construct_object(child, deep=deep))
            if child.comment:
                seqtyp._yaml_add_comment(child.comment, key=idx)
            seqtyp._yaml_set_idx_line_col(idx, [child.start_mark.line, child.start_mark.column])
        return ret_val

    def flatten_mapping(self, node):
        """
        This implements the merge key feature http://yaml.org/type/merge.html
        by inserting keys from the merge dict/list of dicts if not yet
        available in this node
        """

        def constructed(value_node):
            if value_node in self.constructed_objects:
                value = self.constructed_objects[value_node]
            else:
                value = self.construct_object(value_node, deep=False)
            return value
        merge_map_list = []
        index = 0
        while index < len(node.value):
            key_node, value_node = node.value[index]
            if key_node.tag == u'tag:yaml.org,2002:merge':
                if merge_map_list:
                    if self.allow_duplicate_keys:
                        del node.value[index]
                        index += 1
                        continue
                    args = ['while constructing a mapping', node.start_mark, 'found duplicate key "{}"'.format(key_node.value), key_node.start_mark, '\n                        To suppress this check see:\n                           http://yaml.readthedocs.io/en/latest/api.html#duplicate-keys\n                        ', '                        Duplicate keys will become an error in future releases, and are errors\n                        by default when using the new API.\n                        ']
                    if self.allow_duplicate_keys is None:
                        warnings.warn(DuplicateKeyFutureWarning(*args))
                    else:
                        raise DuplicateKeyError(*args)
                del node.value[index]
                if isinstance(value_node, MappingNode):
                    merge_map_list.append((index, constructed(value_node)))
                elif isinstance(value_node, SequenceNode):
                    for subnode in value_node.value:
                        if not isinstance(subnode, MappingNode):
                            raise ConstructorError('while constructing a mapping', node.start_mark, 'expected a mapping for merging, but found %s' % subnode.id, subnode.start_mark)
                        merge_map_list.append((index, constructed(subnode)))
                else:
                    raise ConstructorError('while constructing a mapping', node.start_mark, 'expected a mapping or list of mappings for merging, but found %s' % value_node.id, value_node.start_mark)
            elif key_node.tag == u'tag:yaml.org,2002:value':
                key_node.tag = u'tag:yaml.org,2002:str'
                index += 1
            else:
                index += 1
        return merge_map_list

    def _sentinel(self):
        pass

    def construct_mapping(self, node, maptyp, deep=False):
        if not isinstance(node, MappingNode):
            raise ConstructorError(None, None, 'expected a mapping node, but found %s' % node.id, node.start_mark)
        merge_map = self.flatten_mapping(node)
        if node.comment:
            maptyp._yaml_add_comment(node.comment[:2])
            if len(node.comment) > 2:
                maptyp.yaml_end_comment_extend(node.comment[2], clear=True)
        if node.anchor:
            from ruamel.yaml.serializer import templated_id
            if not templated_id(node.anchor):
                maptyp.yaml_set_anchor(node.anchor)
        last_key, last_value = (None, self._sentinel)
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=True)
            if not isinstance(key, Hashable):
                if isinstance(key, MutableSequence):
                    key_s = CommentedKeySeq(key)
                    if key_node.flow_style is True:
                        key_s.fa.set_flow_style()
                    elif key_node.flow_style is False:
                        key_s.fa.set_block_style()
                    key = key_s
                elif isinstance(key, MutableMapping):
                    key_m = CommentedKeyMap(key)
                    if key_node.flow_style is True:
                        key_m.fa.set_flow_style()
                    elif key_node.flow_style is False:
                        key_m.fa.set_block_style()
                    key = key_m
            if PY2:
                try:
                    hash(key)
                except TypeError as exc:
                    raise ConstructorError('while constructing a mapping', node.start_mark, 'found unacceptable key (%s)' % exc, key_node.start_mark)
            elif not isinstance(key, Hashable):
                raise ConstructorError('while constructing a mapping', node.start_mark, 'found unhashable key', key_node.start_mark)
            value = self.construct_object(value_node, deep=deep)
            if self.check_mapping_key(node, key_node, maptyp, key, value):
                if key_node.comment and len(key_node.comment) > 4 and key_node.comment[4]:
                    if last_value is None:
                        key_node.comment[0] = key_node.comment.pop(4)
                        maptyp._yaml_add_comment(key_node.comment, value=last_key)
                    else:
                        key_node.comment[2] = key_node.comment.pop(4)
                        maptyp._yaml_add_comment(key_node.comment, key=key)
                    key_node.comment = None
                if key_node.comment:
                    maptyp._yaml_add_comment(key_node.comment, key=key)
                if value_node.comment:
                    maptyp._yaml_add_comment(value_node.comment, value=key)
                maptyp._yaml_set_kv_line_col(key, [key_node.start_mark.line, key_node.start_mark.column, value_node.start_mark.line, value_node.start_mark.column])
                maptyp[key] = value
                last_key, last_value = (key, value)
        if merge_map:
            maptyp.add_yaml_merge(merge_map)

    def construct_setting(self, node, typ, deep=False):
        if not isinstance(node, MappingNode):
            raise ConstructorError(None, None, 'expected a mapping node, but found %s' % node.id, node.start_mark)
        if node.comment:
            typ._yaml_add_comment(node.comment[:2])
            if len(node.comment) > 2:
                typ.yaml_end_comment_extend(node.comment[2], clear=True)
        if node.anchor:
            from ruamel.yaml.serializer import templated_id
            if not templated_id(node.anchor):
                typ.yaml_set_anchor(node.anchor)
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=True)
            if not isinstance(key, Hashable):
                if isinstance(key, list):
                    key = tuple(key)
            if PY2:
                try:
                    hash(key)
                except TypeError as exc:
                    raise ConstructorError('while constructing a mapping', node.start_mark, 'found unacceptable key (%s)' % exc, key_node.start_mark)
            elif not isinstance(key, Hashable):
                raise ConstructorError('while constructing a mapping', node.start_mark, 'found unhashable key', key_node.start_mark)
            value = self.construct_object(value_node, deep=deep)
            self.check_set_key(node, key_node, typ, key)
            if key_node.comment:
                typ._yaml_add_comment(key_node.comment, key=key)
            if value_node.comment:
                typ._yaml_add_comment(value_node.comment, value=key)
            typ.add(key)

    def construct_yaml_seq(self, node):
        data = CommentedSeq()
        data._yaml_set_line_col(node.start_mark.line, node.start_mark.column)
        if node.comment:
            data._yaml_add_comment(node.comment)
        yield data
        data.extend(self.construct_rt_sequence(node, data))
        self.set_collection_style(data, node)

    def construct_yaml_map(self, node):
        data = CommentedMap()
        data._yaml_set_line_col(node.start_mark.line, node.start_mark.column)
        yield data
        self.construct_mapping(node, data, deep=True)
        self.set_collection_style(data, node)

    def set_collection_style(self, data, node):
        if len(data) == 0:
            return
        if node.flow_style is True:
            data.fa.set_flow_style()
        elif node.flow_style is False:
            data.fa.set_block_style()

    def construct_yaml_object(self, node, cls):
        data = cls.__new__(cls)
        yield data
        if hasattr(data, '__setstate__'):
            state = SafeConstructor.construct_mapping(self, node, deep=True)
            data.__setstate__(state)
        else:
            state = SafeConstructor.construct_mapping(self, node)
            data.__dict__.update(state)

    def construct_yaml_omap(self, node):
        omap = CommentedOrderedMap()
        omap._yaml_set_line_col(node.start_mark.line, node.start_mark.column)
        if node.flow_style is True:
            omap.fa.set_flow_style()
        elif node.flow_style is False:
            omap.fa.set_block_style()
        yield omap
        if node.comment:
            omap._yaml_add_comment(node.comment[:2])
            if len(node.comment) > 2:
                omap.yaml_end_comment_extend(node.comment[2], clear=True)
        if not isinstance(node, SequenceNode):
            raise ConstructorError('while constructing an ordered map', node.start_mark, 'expected a sequence, but found %s' % node.id, node.start_mark)
        for subnode in node.value:
            if not isinstance(subnode, MappingNode):
                raise ConstructorError('while constructing an ordered map', node.start_mark, 'expected a mapping of length 1, but found %s' % subnode.id, subnode.start_mark)
            if len(subnode.value) != 1:
                raise ConstructorError('while constructing an ordered map', node.start_mark, 'expected a single mapping item, but found %d items' % len(subnode.value), subnode.start_mark)
            key_node, value_node = subnode.value[0]
            key = self.construct_object(key_node)
            assert key not in omap
            value = self.construct_object(value_node)
            if key_node.comment:
                omap._yaml_add_comment(key_node.comment, key=key)
            if subnode.comment:
                omap._yaml_add_comment(subnode.comment, key=key)
            if value_node.comment:
                omap._yaml_add_comment(value_node.comment, value=key)
            omap[key] = value

    def construct_yaml_set(self, node):
        data = CommentedSet()
        data._yaml_set_line_col(node.start_mark.line, node.start_mark.column)
        yield data
        self.construct_setting(node, data)

    def construct_undefined(self, node):
        try:
            if isinstance(node, MappingNode):
                data = CommentedMap()
                data._yaml_set_line_col(node.start_mark.line, node.start_mark.column)
                if node.flow_style is True:
                    data.fa.set_flow_style()
                elif node.flow_style is False:
                    data.fa.set_block_style()
                data.yaml_set_tag(node.tag)
                yield data
                if node.anchor:
                    data.yaml_set_anchor(node.anchor)
                self.construct_mapping(node, data)
                return
            elif isinstance(node, ScalarNode):
                data2 = TaggedScalar()
                data2.value = self.construct_scalar(node)
                data2.style = node.style
                data2.yaml_set_tag(node.tag)
                yield data2
                if node.anchor:
                    data2.yaml_set_anchor(node.anchor, always_dump=True)
                return
            elif isinstance(node, SequenceNode):
                data3 = CommentedSeq()
                data3._yaml_set_line_col(node.start_mark.line, node.start_mark.column)
                if node.flow_style is True:
                    data3.fa.set_flow_style()
                elif node.flow_style is False:
                    data3.fa.set_block_style()
                data3.yaml_set_tag(node.tag)
                yield data3
                if node.anchor:
                    data3.yaml_set_anchor(node.anchor)
                data3.extend(self.construct_sequence(node))
                return
        except:
            pass
        raise ConstructorError(None, None, 'could not determine a constructor for the tag %r' % utf8(node.tag), node.start_mark)

    def construct_yaml_timestamp(self, node, values=None):
        try:
            match = self.timestamp_regexp.match(node.value)
        except TypeError:
            match = None
        if match is None:
            raise ConstructorError(None, None, 'failed to construct timestamp from "{}"'.format(node.value), node.start_mark)
        values = match.groupdict()
        if not values['hour']:
            return SafeConstructor.construct_yaml_timestamp(self, node, values)
        for part in ['t', 'tz_sign', 'tz_hour', 'tz_minute']:
            if values[part]:
                break
        else:
            return SafeConstructor.construct_yaml_timestamp(self, node, values)
        year = int(values['year'])
        month = int(values['month'])
        day = int(values['day'])
        hour = int(values['hour'])
        minute = int(values['minute'])
        second = int(values['second'])
        fraction = 0
        if values['fraction']:
            fraction_s = values['fraction'][:6]
            while len(fraction_s) < 6:
                fraction_s += '0'
            fraction = int(fraction_s)
            if len(values['fraction']) > 6 and int(values['fraction'][6]) > 4:
                fraction += 1
        delta = None
        if values['tz_sign']:
            tz_hour = int(values['tz_hour'])
            minutes = values['tz_minute']
            tz_minute = int(minutes) if minutes else 0
            delta = datetime.timedelta(hours=tz_hour, minutes=tz_minute)
            if values['tz_sign'] == '-':
                delta = -delta
        if delta:
            dt = datetime.datetime(year, month, day, hour, minute)
            dt -= delta
            data = TimeStamp(dt.year, dt.month, dt.day, dt.hour, dt.minute, second, fraction)
            data._yaml['delta'] = delta
            tz = values['tz_sign'] + values['tz_hour']
            if values['tz_minute']:
                tz += ':' + values['tz_minute']
            data._yaml['tz'] = tz
        else:
            data = TimeStamp(year, month, day, hour, minute, second, fraction)
            if values['tz']:
                data._yaml['tz'] = values['tz']
        if values['t']:
            data._yaml['t'] = True
        return data

    def construct_yaml_bool(self, node):
        b = SafeConstructor.construct_yaml_bool(self, node)
        if node.anchor:
            return ScalarBoolean(b, anchor=node.anchor)
        return b