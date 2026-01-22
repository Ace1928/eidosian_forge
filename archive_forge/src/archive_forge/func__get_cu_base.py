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
def _get_cu_base(cu):
    top_die = cu.get_top_DIE()
    attr = top_die.attributes
    if 'DW_AT_low_pc' in attr:
        return attr['DW_AT_low_pc'].value
    elif 'DW_AT_entry_pc' in attr:
        return attr['DW_AT_entry_pc'].value
    elif 'DW_AT_ranges' in attr:
        rl = cu.dwarfinfo.range_lists().get_range_list_at_offset(attr['DW_AT_ranges'].value, cu)
        base_ip = None
        for r in rl:
            if isinstance(r, RangeBaseAddressEntry):
                ip = r.base_address
            elif isinstance(r, RangeEntry) and r.is_absolute:
                ip = r.begin_offset
            else:
                ip = None
            if ip is not None and (base_ip is None or ip < base_ip):
                base_ip = ip
        if base_ip is None:
            raise ValueError("Can't find the base IP (low_pc) for a CU")
        return base_ip
    else:
        raise ValueError("Can't find the base IP (low_pc) for a CU")