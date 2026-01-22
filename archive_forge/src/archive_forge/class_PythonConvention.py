import abc
import re
class PythonConvention(Convention):

    def convert_function_name(self, name):
        return name

    def convert_parameter_name(self, name):
        return name