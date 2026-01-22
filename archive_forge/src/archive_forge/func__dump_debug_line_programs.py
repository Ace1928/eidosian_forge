import argparse
import os, sys
import re
import string
import traceback
import itertools
from elftools import __version__
from elftools.common.exceptions import ELFError
from elftools.common.utils import bytes2str, iterbytes
from elftools.elf.elffile import ELFFile
from elftools.elf.dynamic import DynamicSection, DynamicSegment
from elftools.elf.enums import ENUM_D_TAG
from elftools.elf.segments import InterpSegment
from elftools.elf.sections import (
from elftools.elf.gnuversions import (
from elftools.elf.relocation import RelocationSection
from elftools.elf.descriptions import (
from elftools.elf.constants import E_FLAGS
from elftools.elf.constants import E_FLAGS_MASKS
from elftools.elf.constants import SH_FLAGS
from elftools.elf.constants import SHN_INDICES
from elftools.dwarf.dwarfinfo import DWARFInfo
from elftools.dwarf.descriptions import (
from elftools.dwarf.constants import (
from elftools.dwarf.locationlists import LocationParser, LocationEntry, LocationViewPair, BaseAddressEntry as LocBaseAddressEntry, LocationListsPair
from elftools.dwarf.ranges import RangeEntry, BaseAddressEntry as RangeBaseAddressEntry, RangeListsPair
from elftools.dwarf.callframe import CIE, FDE, ZERO
from elftools.ehabi.ehabiinfo import CorruptEHABIEntry, CannotUnwindEHABIEntry, GenericEHABIEntry
from elftools.dwarf.enums import ENUM_DW_UT
def _dump_debug_line_programs(self):
    """ Dump the (decoded) line programs from .debug_line
            The programs are dumped in the order of the CUs they belong to.
        """
    if not self._dwarfinfo.has_debug_info:
        return
    self._emitline('Contents of the %s section:' % self._dwarfinfo.debug_line_sec.name)
    self._emitline()
    lineprogram_list = []
    for cu in self._dwarfinfo.iter_CUs():
        lineprogram = self._dwarfinfo.line_program_for_CU(cu)
        if lineprogram in lineprogram_list:
            continue
        lineprogram_list.append(lineprogram)
        ver5 = lineprogram.header.version >= 5
        cu_filename = bytes2str(lineprogram['file_entry'][0].name)
        if len(lineprogram['include_directory']) > 0:
            self._emitline('%s:' % cu_filename)
        else:
            self._emitline('CU: %s:' % cu_filename)
        self._emitline('File name                            Line number    Starting address    Stmt')
        for entry in lineprogram.get_entries():
            state = entry.state
            if state is None:
                if entry.command == DW_LNS_set_file:
                    file_entry = lineprogram['file_entry'][entry.args[0] - 1]
                    if file_entry.dir_index == 0:
                        self._emitline('\n./%s:[++]' % bytes2str(file_entry.name))
                    else:
                        self._emitline('\n%s/%s:' % (bytes2str(lineprogram['include_directory'][file_entry.dir_index - 1]), bytes2str(file_entry.name)))
                elif entry.command == DW_LNE_define_file:
                    self._emitline('%s:' % bytes2str(lineprogram['include_directory'][entry.args[0].dir_index]))
            elif lineprogram['version'] < 4 or self.elffile['e_machine'] == 'EM_PPC64':
                self._emitline('%-35s  %11s  %18s    %s' % (bytes2str(lineprogram['file_entry'][state.file - 1].name), state.line if not state.end_sequence else '-', '0' if state.address == 0 else self._format_hex(state.address), 'x' if state.is_stmt and (not state.end_sequence) else ''))
            else:
                self._emitline('%-35s  %s  %18s%s %s' % (bytes2str(lineprogram['file_entry'][state.file - 1].name), '%11d' % (state.line,) if not state.end_sequence else '-', '0' if state.address == 0 else self._format_hex(state.address), '' if lineprogram.header.maximum_operations_per_instruction == 1 else '[%d]' % (state.op_index,), 'x' if state.is_stmt and (not state.end_sequence) else ''))
            if entry.command == DW_LNS_copy:
                self._emitline()