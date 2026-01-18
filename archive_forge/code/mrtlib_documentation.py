import abc
import logging
import struct
import time
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib import stringify
from os_ken.lib import type_desc
from os_ken.lib.packet import bgp
from os_ken.lib.packet import ospf

    MRT format file writer.

    ========= ================================================
    Argument  Description
    ========= ================================================
    f         File object which writing MRT format file
              in binary mode.
    ========= ================================================

    Example of usage::

        import bz2
        import time
        from os_ken.lib import mrtlib
        from os_ken.lib.packet import bgp

        mrt_writer = mrtlib.Writer(
            bz2.BZ2File('rib.YYYYMMDD.hhmm.bz2', 'wb'))

        prefix = bgp.IPAddrPrefix(24, '10.0.0.0')

        rib_entry = mrtlib.MrtRibEntry(
            peer_index=0,
            originated_time=int(time.time()),
            bgp_attributes=[bgp.BGPPathAttributeOrigin(0)])

        message = mrtlib.TableDump2RibIPv4UnicastMrtMessage(
            seq_num=0,
            prefix=prefix,
            rib_entries=[rib_entry])

        record = mrtlib.TableDump2MrtRecord(
            message=message)

        mrt_writer.write(record)
    