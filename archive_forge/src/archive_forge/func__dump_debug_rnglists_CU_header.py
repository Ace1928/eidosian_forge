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
def _dump_debug_rnglists_CU_header(self, cu):
    self._emitline(' Table at Offset: %s:' % self._format_hex(cu.cu_offset, alternate=True))
    self._emitline('  Length:          %s' % self._format_hex(cu.unit_length, alternate=True))
    self._emitline('  DWARF version:   %d' % cu.version)
    self._emitline('  Address size:    %d' % cu.address_size)
    self._emitline('  Segment size:    %d' % cu.segment_selector_size)
    self._emitline('  Offset entries:  %d\n' % cu.offset_count)
    if cu.offsets and len(cu.offsets):
        self._emitline('  Offsets starting at 0x%x:' % cu.offset_table_offset)
        for i_offset in enumerate(cu.offsets):
            self._emitline('    [%6d] 0x%x' % i_offset)