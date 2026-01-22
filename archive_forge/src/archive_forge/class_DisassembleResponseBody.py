from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class DisassembleResponseBody(BaseSchema):
    """
    "body" of DisassembleResponse

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'instructions': {'type': 'array', 'items': {'$ref': '#/definitions/DisassembledInstruction'}, 'description': 'The list of disassembled instructions.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, instructions, update_ids_from_dap=False, **kwargs):
        """
        :param array instructions: The list of disassembled instructions.
        """
        self.instructions = instructions
        if update_ids_from_dap and self.instructions:
            for o in self.instructions:
                DisassembledInstruction.update_dict_ids_from_dap(o)
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        instructions = self.instructions
        if instructions and hasattr(instructions[0], 'to_dict'):
            instructions = [x.to_dict() for x in instructions]
        dct = {'instructions': [DisassembledInstruction.update_dict_ids_to_dap(o) for o in instructions] if update_ids_to_dap and instructions else instructions}
        dct.update(self.kwargs)
        return dct