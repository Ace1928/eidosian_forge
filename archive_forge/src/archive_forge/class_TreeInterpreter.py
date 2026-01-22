import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
class TreeInterpreter(Visitor):
    COMPARATOR_FUNC = {'eq': _equals, 'ne': lambda x, y: not _equals(x, y), 'lt': operator.lt, 'gt': operator.gt, 'lte': operator.le, 'gte': operator.ge}
    _EQUALITY_OPS = ['eq', 'ne']
    MAP_TYPE = dict

    def __init__(self, options=None):
        super(TreeInterpreter, self).__init__()
        self._dict_cls = self.MAP_TYPE
        if options is None:
            options = Options()
        self._options = options
        if options.dict_cls is not None:
            self._dict_cls = self._options.dict_cls
        if options.custom_functions is not None:
            self._functions = self._options.custom_functions
        else:
            self._functions = functions.Functions()

    def default_visit(self, node, *args, **kwargs):
        raise NotImplementedError(node['type'])

    def visit_subexpression(self, node, value):
        result = value
        for node in node['children']:
            result = self.visit(node, result)
        return result

    def visit_field(self, node, value):
        try:
            return value.get(node['value'])
        except AttributeError:
            return None

    def visit_comparator(self, node, value):
        comparator_func = self.COMPARATOR_FUNC[node['value']]
        if node['value'] in self._EQUALITY_OPS:
            return comparator_func(self.visit(node['children'][0], value), self.visit(node['children'][1], value))
        else:
            left = self.visit(node['children'][0], value)
            right = self.visit(node['children'][1], value)
            num_types = (int, float)
            if not (_is_comparable(left) and _is_comparable(right)):
                return None
            return comparator_func(left, right)

    def visit_current(self, node, value):
        return value

    def visit_expref(self, node, value):
        return _Expression(node['children'][0], self)

    def visit_function_expression(self, node, value):
        resolved_args = []
        for child in node['children']:
            current = self.visit(child, value)
            resolved_args.append(current)
        return self._functions.call_function(node['value'], resolved_args)

    def visit_filter_projection(self, node, value):
        base = self.visit(node['children'][0], value)
        if not isinstance(base, list):
            return None
        comparator_node = node['children'][2]
        collected = []
        for element in base:
            if self._is_true(self.visit(comparator_node, element)):
                current = self.visit(node['children'][1], element)
                if current is not None:
                    collected.append(current)
        return collected

    def visit_flatten(self, node, value):
        base = self.visit(node['children'][0], value)
        if not isinstance(base, list):
            return None
        merged_list = []
        for element in base:
            if isinstance(element, list):
                merged_list.extend(element)
            else:
                merged_list.append(element)
        return merged_list

    def visit_identity(self, node, value):
        return value

    def visit_index(self, node, value):
        if not isinstance(value, list):
            return None
        try:
            return value[node['value']]
        except IndexError:
            return None

    def visit_index_expression(self, node, value):
        result = value
        for node in node['children']:
            result = self.visit(node, result)
        return result

    def visit_slice(self, node, value):
        if not isinstance(value, list):
            return None
        s = slice(*node['children'])
        return value[s]

    def visit_key_val_pair(self, node, value):
        return self.visit(node['children'][0], value)

    def visit_literal(self, node, value):
        return node['value']

    def visit_multi_select_dict(self, node, value):
        if value is None:
            return None
        collected = self._dict_cls()
        for child in node['children']:
            collected[child['value']] = self.visit(child, value)
        return collected

    def visit_multi_select_list(self, node, value):
        if value is None:
            return None
        collected = []
        for child in node['children']:
            collected.append(self.visit(child, value))
        return collected

    def visit_or_expression(self, node, value):
        matched = self.visit(node['children'][0], value)
        if self._is_false(matched):
            matched = self.visit(node['children'][1], value)
        return matched

    def visit_and_expression(self, node, value):
        matched = self.visit(node['children'][0], value)
        if self._is_false(matched):
            return matched
        return self.visit(node['children'][1], value)

    def visit_not_expression(self, node, value):
        original_result = self.visit(node['children'][0], value)
        if type(original_result) is int and original_result == 0:
            return False
        return not original_result

    def visit_pipe(self, node, value):
        result = value
        for node in node['children']:
            result = self.visit(node, result)
        return result

    def visit_projection(self, node, value):
        base = self.visit(node['children'][0], value)
        if not isinstance(base, list):
            return None
        collected = []
        for element in base:
            current = self.visit(node['children'][1], element)
            if current is not None:
                collected.append(current)
        return collected

    def visit_value_projection(self, node, value):
        base = self.visit(node['children'][0], value)
        try:
            base = base.values()
        except AttributeError:
            return None
        collected = []
        for element in base:
            current = self.visit(node['children'][1], element)
            if current is not None:
                collected.append(current)
        return collected

    def _is_false(self, value):
        return value == '' or value == [] or value == {} or (value is None) or (value is False)

    def _is_true(self, value):
        return not self._is_false(value)