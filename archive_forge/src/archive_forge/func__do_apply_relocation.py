from collections import namedtuple
from ..common.exceptions import ELFRelocationError
from ..common.utils import elf_assert, struct_parse
from .sections import Section
from .enums import (
from ..construct import Container
def _do_apply_relocation(self, stream, reloc, symtab):
    if reloc['r_info_sym'] >= symtab.num_symbols():
        raise ELFRelocationError('Invalid symbol reference in relocation: index %s' % reloc['r_info_sym'])
    sym_value = symtab.get_symbol(reloc['r_info_sym'])['st_value']
    reloc_type = reloc['r_info_type']
    recipe = None
    if self.elffile.get_machine_arch() == 'x86':
        if reloc.is_RELA():
            raise ELFRelocationError('Unexpected RELA relocation for x86: %s' % reloc)
        recipe = self._RELOCATION_RECIPES_X86.get(reloc_type, None)
    elif self.elffile.get_machine_arch() == 'x64':
        if not reloc.is_RELA():
            raise ELFRelocationError('Unexpected REL relocation for x64: %s' % reloc)
        recipe = self._RELOCATION_RECIPES_X64.get(reloc_type, None)
    elif self.elffile.get_machine_arch() == 'MIPS':
        if reloc.is_RELA():
            if reloc_type == ENUM_RELOC_TYPE_MIPS['R_MIPS_64']:
                if reloc['r_type2'] != 0 or reloc['r_type3'] != 0 or reloc['r_ssym'] != 0:
                    raise ELFRelocationError('Multiple relocations in R_MIPS_64 are not implemented: %s' % reloc)
            recipe = self._RELOCATION_RECIPES_MIPS_RELA.get(reloc_type, None)
        else:
            recipe = self._RELOCATION_RECIPES_MIPS_REL.get(reloc_type, None)
    elif self.elffile.get_machine_arch() == 'ARM':
        if reloc.is_RELA():
            raise ELFRelocationError('Unexpected RELA relocation for ARM: %s' % reloc)
        recipe = self._RELOCATION_RECIPES_ARM.get(reloc_type, None)
    elif self.elffile.get_machine_arch() == 'AArch64':
        recipe = self._RELOCATION_RECIPES_AARCH64.get(reloc_type, None)
    elif self.elffile.get_machine_arch() == '64-bit PowerPC':
        recipe = self._RELOCATION_RECIPES_PPC64.get(reloc_type, None)
    elif self.elffile.get_machine_arch() == 'IBM S/390':
        recipe = self._RELOCATION_RECIPES_S390X.get(reloc_type, None)
    elif self.elffile.get_machine_arch() == 'Linux BPF - in-kernel virtual machine':
        recipe = self._RELOCATION_RECIPES_EBPF.get(reloc_type, None)
    elif self.elffile.get_machine_arch() == 'LoongArch':
        if not reloc.is_RELA():
            raise ELFRelocationError('Unexpected REL relocation for LoongArch: %s' % reloc)
        recipe = self._RELOCATION_RECIPES_LOONGARCH.get(reloc_type, None)
    if recipe is None:
        raise ELFRelocationError('Unsupported relocation type: %s' % reloc_type)
    if recipe.bytesize == 4:
        value_struct = self.elffile.structs.Elf_word('')
    elif recipe.bytesize == 8:
        value_struct = self.elffile.structs.Elf_word64('')
    elif recipe.bytesize == 1:
        value_struct = self.elffile.structs.Elf_byte('')
    elif recipe.bytesize == 2:
        value_struct = self.elffile.structs.Elf_half('')
    else:
        raise ELFRelocationError('Invalid bytesize %s for relocation' % recipe.bytesize)
    original_value = struct_parse(value_struct, stream, stream_pos=reloc['r_offset'])
    relocated_value = recipe.calc_func(value=original_value, sym_value=sym_value, offset=reloc['r_offset'], addend=reloc['r_addend'] if recipe.has_addend else 0)
    stream.seek(reloc['r_offset'])
    relocated_value = relocated_value % 2 ** (recipe.bytesize * 8)
    value_struct.build_stream(relocated_value, stream)