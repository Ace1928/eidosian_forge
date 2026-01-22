from ..common.utils import struct_parse
from .decoder import EHABIBytecodeDecoder
from .constants import EHABI_INDEX_ENTRY_SIZE
from .structs import EHABIStructs
class EHABIInfo(object):
    """ ARM exception handler abi information class.

        Parameters:

            arm_idx_section:
                elf.sections.Section object, section which type is SHT_ARM_EXIDX.

            little_endian:
                bool, endianness of elf file.
    """

    def __init__(self, arm_idx_section, little_endian):
        self._arm_idx_section = arm_idx_section
        self._struct = EHABIStructs(little_endian)
        self._num_entry = None

    def section_name(self):
        return self._arm_idx_section.name

    def section_offset(self):
        return self._arm_idx_section['sh_offset']

    def num_entry(self):
        """ Number of exception handler entry in the section.
        """
        if self._num_entry is None:
            self._num_entry = self._arm_idx_section['sh_size'] // EHABI_INDEX_ENTRY_SIZE
        return self._num_entry

    def get_entry(self, n):
        """ Get the exception handler entry at index #n. (EHABIEntry object or a subclass)
        """
        if n >= self.num_entry():
            raise IndexError('Invalid entry %d/%d' % (n, self._num_entry))
        eh_index_entry_offset = self.section_offset() + n * EHABI_INDEX_ENTRY_SIZE
        eh_index_data = struct_parse(self._struct.EH_index_struct, self._arm_idx_section.stream, eh_index_entry_offset)
        word0, word1 = (eh_index_data['word0'], eh_index_data['word1'])
        if word0 & 2147483648 != 0:
            return CorruptEHABIEntry('Corrupt ARM exception handler table entry: %x' % n)
        function_offset = arm_expand_prel31(word0, self.section_offset() + n * EHABI_INDEX_ENTRY_SIZE)
        if word1 == 1:
            return CannotUnwindEHABIEntry(function_offset)
        elif word1 & 2147483648 == 0:
            eh_table_offset = arm_expand_prel31(word1, self.section_offset() + n * EHABI_INDEX_ENTRY_SIZE + 4)
            eh_index_data = struct_parse(self._struct.EH_table_struct, self._arm_idx_section.stream, eh_table_offset)
            word0 = eh_index_data['word0']
            if word0 & 2147483648 == 0:
                return GenericEHABIEntry(function_offset, arm_expand_prel31(word0, eh_table_offset))
            else:
                if word0 & 1879048192 != 0:
                    return CorruptEHABIEntry('Corrupt ARM compact model table entry: %x' % n)
                per_index = word0 >> 24 & 127
                if per_index == 0:
                    opcode = [(word0 & 16711680) >> 16, (word0 & 65280) >> 8, word0 & 255]
                    return EHABIEntry(function_offset, per_index, opcode)
                elif per_index == 1 or per_index == 2:
                    more_word = word0 >> 16 & 255
                    opcode = [word0 >> 8 & 255, word0 >> 0 & 255]
                    self._arm_idx_section.stream.seek(eh_table_offset + 4)
                    for i in range(more_word):
                        r = struct_parse(self._struct.EH_table_struct, self._arm_idx_section.stream)['word0']
                        opcode.append(r >> 24 & 255)
                        opcode.append(r >> 16 & 255)
                        opcode.append(r >> 8 & 255)
                        opcode.append(r >> 0 & 255)
                    return EHABIEntry(function_offset, per_index, opcode, eh_table_offset=eh_table_offset)
                else:
                    return CorruptEHABIEntry('Unknown ARM compact model %d at table entry: %x' % (per_index, n))
        else:
            if word1 & 2130706432 != 0:
                return CorruptEHABIEntry('Corrupt ARM compact model table entry: %x' % n)
            opcode = [(word1 & 16711680) >> 16, (word1 & 65280) >> 8, word1 & 255]
            return EHABIEntry(function_offset, 0, opcode)