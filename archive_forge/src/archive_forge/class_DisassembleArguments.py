from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class DisassembleArguments(BaseSchema):
    """
    Arguments for 'disassemble' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'memoryReference': {'type': 'string', 'description': 'Memory reference to the base location containing the instructions to disassemble.'}, 'offset': {'type': 'integer', 'description': 'Optional offset (in bytes) to be applied to the reference location before disassembling. Can be negative.'}, 'instructionOffset': {'type': 'integer', 'description': 'Optional offset (in instructions) to be applied after the byte offset (if any) before disassembling. Can be negative.'}, 'instructionCount': {'type': 'integer', 'description': "Number of instructions to disassemble starting at the specified location and offset.\nAn adapter must return exactly this number of instructions - any unavailable instructions should be replaced with an implementation-defined 'invalid instruction' value."}, 'resolveSymbols': {'type': 'boolean', 'description': 'If true, the adapter should attempt to resolve memory addresses and other values to symbolic names.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, memoryReference, instructionCount, offset=None, instructionOffset=None, resolveSymbols=None, update_ids_from_dap=False, **kwargs):
        """
        :param string memoryReference: Memory reference to the base location containing the instructions to disassemble.
        :param integer instructionCount: Number of instructions to disassemble starting at the specified location and offset.
        An adapter must return exactly this number of instructions - any unavailable instructions should be replaced with an implementation-defined 'invalid instruction' value.
        :param integer offset: Optional offset (in bytes) to be applied to the reference location before disassembling. Can be negative.
        :param integer instructionOffset: Optional offset (in instructions) to be applied after the byte offset (if any) before disassembling. Can be negative.
        :param boolean resolveSymbols: If true, the adapter should attempt to resolve memory addresses and other values to symbolic names.
        """
        self.memoryReference = memoryReference
        self.instructionCount = instructionCount
        self.offset = offset
        self.instructionOffset = instructionOffset
        self.resolveSymbols = resolveSymbols
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        memoryReference = self.memoryReference
        instructionCount = self.instructionCount
        offset = self.offset
        instructionOffset = self.instructionOffset
        resolveSymbols = self.resolveSymbols
        dct = {'memoryReference': memoryReference, 'instructionCount': instructionCount}
        if offset is not None:
            dct['offset'] = offset
        if instructionOffset is not None:
            dct['instructionOffset'] = instructionOffset
        if resolveSymbols is not None:
            dct['resolveSymbols'] = resolveSymbols
        dct.update(self.kwargs)
        return dct