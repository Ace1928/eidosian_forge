from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class DisassembledInstruction(BaseSchema):
    """
    Represents a single disassembled instruction.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'address': {'type': 'string', 'description': "The address of the instruction. Treated as a hex value if prefixed with '0x', or as a decimal value otherwise."}, 'instructionBytes': {'type': 'string', 'description': 'Optional raw bytes representing the instruction and its operands, in an implementation-defined format.'}, 'instruction': {'type': 'string', 'description': 'Text representing the instruction and its operands, in an implementation-defined format.'}, 'symbol': {'type': 'string', 'description': 'Name of the symbol that corresponds with the location of this instruction, if any.'}, 'location': {'description': 'Source location that corresponds to this instruction, if any.\nShould always be set (if available) on the first instruction returned,\nbut can be omitted afterwards if this instruction maps to the same source file as the previous instruction.', 'type': 'Source'}, 'line': {'type': 'integer', 'description': 'The line within the source location that corresponds to this instruction, if any.'}, 'column': {'type': 'integer', 'description': 'The column within the line that corresponds to this instruction, if any.'}, 'endLine': {'type': 'integer', 'description': 'The end line of the range that corresponds to this instruction, if any.'}, 'endColumn': {'type': 'integer', 'description': 'The end column of the range that corresponds to this instruction, if any.'}}
    __refs__ = set(['location'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, address, instruction, instructionBytes=None, symbol=None, location=None, line=None, column=None, endLine=None, endColumn=None, update_ids_from_dap=False, **kwargs):
        """
        :param string address: The address of the instruction. Treated as a hex value if prefixed with '0x', or as a decimal value otherwise.
        :param string instruction: Text representing the instruction and its operands, in an implementation-defined format.
        :param string instructionBytes: Optional raw bytes representing the instruction and its operands, in an implementation-defined format.
        :param string symbol: Name of the symbol that corresponds with the location of this instruction, if any.
        :param Source location: Source location that corresponds to this instruction, if any.
        Should always be set (if available) on the first instruction returned,
        but can be omitted afterwards if this instruction maps to the same source file as the previous instruction.
        :param integer line: The line within the source location that corresponds to this instruction, if any.
        :param integer column: The column within the line that corresponds to this instruction, if any.
        :param integer endLine: The end line of the range that corresponds to this instruction, if any.
        :param integer endColumn: The end column of the range that corresponds to this instruction, if any.
        """
        self.address = address
        self.instruction = instruction
        self.instructionBytes = instructionBytes
        self.symbol = symbol
        if location is None:
            self.location = Source()
        else:
            self.location = Source(update_ids_from_dap=update_ids_from_dap, **location) if location.__class__ != Source else location
        self.line = line
        self.column = column
        self.endLine = endLine
        self.endColumn = endColumn
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        address = self.address
        instruction = self.instruction
        instructionBytes = self.instructionBytes
        symbol = self.symbol
        location = self.location
        line = self.line
        column = self.column
        endLine = self.endLine
        endColumn = self.endColumn
        dct = {'address': address, 'instruction': instruction}
        if instructionBytes is not None:
            dct['instructionBytes'] = instructionBytes
        if symbol is not None:
            dct['symbol'] = symbol
        if location is not None:
            dct['location'] = location.to_dict(update_ids_to_dap=update_ids_to_dap)
        if line is not None:
            dct['line'] = line
        if column is not None:
            dct['column'] = column
        if endLine is not None:
            dct['endLine'] = endLine
        if endColumn is not None:
            dct['endColumn'] = endColumn
        dct.update(self.kwargs)
        return dct