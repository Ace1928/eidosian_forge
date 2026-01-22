from ..common.utils import struct_parse, dwarf_assert
class AbbrevTable(object):
    """ Represents a DWARF abbreviation table.
    """

    def __init__(self, structs, stream, offset):
        """ Create new abbreviation table. Parses the actual table from the
            stream and stores it internally.

            structs:
                A DWARFStructs instance for parsing the data

            stream, offset:
                The stream and offset into the stream where this abbreviation
                table lives.
        """
        self.structs = structs
        self.stream = stream
        self.offset = offset
        self._abbrev_map = self._parse_abbrev_table()

    def get_abbrev(self, code):
        """ Get the AbbrevDecl for a given code. Raise KeyError if no
            declaration for this code exists.
        """
        return self._abbrev_map[code]

    def _parse_abbrev_table(self):
        """ Parse the abbrev table from the stream
        """
        map = {}
        self.stream.seek(self.offset)
        while True:
            decl_code = struct_parse(struct=self.structs.Dwarf_uleb128(''), stream=self.stream)
            if decl_code == 0:
                break
            declaration = struct_parse(struct=self.structs.Dwarf_abbrev_declaration, stream=self.stream)
            map[decl_code] = AbbrevDecl(decl_code, declaration)
        return map