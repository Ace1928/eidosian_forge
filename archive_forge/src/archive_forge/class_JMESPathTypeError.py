from jmespath.compat import with_str_method
@with_str_method
class JMESPathTypeError(JMESPathError):

    def __init__(self, function_name, current_value, actual_type, expected_types):
        self.function_name = function_name
        self.current_value = current_value
        self.actual_type = actual_type
        self.expected_types = expected_types

    def __str__(self):
        return 'In function %s(), invalid type for value: %s, expected one of: %s, received: "%s"' % (self.function_name, self.current_value, self.expected_types, self.actual_type)