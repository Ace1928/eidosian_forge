import decimal
from .draft04 import CodeGeneratorDraft04, JSON_TYPE_TO_PYTHON_TYPE
from .exceptions import JsonSchemaDefinitionException
from .generator import enforce_list
def _generate_func_code_block(self, definition):
    if isinstance(definition, bool):
        self.generate_boolean_schema()
    elif '$ref' in definition:
        self.generate_ref()
    else:
        self.run_generate_functions(definition)