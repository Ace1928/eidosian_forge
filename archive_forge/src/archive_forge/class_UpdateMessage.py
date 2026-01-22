from typing import Any, List, Optional, Union
import dns.message
import dns.name
import dns.opcode
import dns.rdata
import dns.rdataclass
import dns.rdataset
import dns.rdatatype
import dns.tsig
class UpdateMessage(dns.message.Message):
    _section_enum = UpdateSection

    def __init__(self, zone: Optional[Union[dns.name.Name, str]]=None, rdclass: dns.rdataclass.RdataClass=dns.rdataclass.IN, keyring: Optional[Any]=None, keyname: Optional[dns.name.Name]=None, keyalgorithm: Union[dns.name.Name, str]=dns.tsig.default_algorithm, id: Optional[int]=None):
        """Initialize a new DNS Update object.

        See the documentation of the Message class for a complete
        description of the keyring dictionary.

        *zone*, a ``dns.name.Name``, ``str``, or ``None``, the zone
        which is being updated.  ``None`` should only be used by dnspython's
        message constructors, as a zone is required for the convenience
        methods like ``add()``, ``replace()``, etc.

        *rdclass*, an ``int`` or ``str``, the class of the zone.

        The *keyring*, *keyname*, and *keyalgorithm* parameters are passed to
        ``use_tsig()``; see its documentation for details.
        """
        super().__init__(id=id)
        self.flags |= dns.opcode.to_flags(dns.opcode.UPDATE)
        if isinstance(zone, str):
            zone = dns.name.from_text(zone)
        self.origin = zone
        rdclass = dns.rdataclass.RdataClass.make(rdclass)
        self.zone_rdclass = rdclass
        if self.origin:
            self.find_rrset(self.zone, self.origin, rdclass, dns.rdatatype.SOA, create=True, force_unique=True)
        if keyring is not None:
            self.use_tsig(keyring, keyname, algorithm=keyalgorithm)

    @property
    def zone(self) -> List[dns.rrset.RRset]:
        """The zone section."""
        return self.sections[0]

    @zone.setter
    def zone(self, v):
        self.sections[0] = v

    @property
    def prerequisite(self) -> List[dns.rrset.RRset]:
        """The prerequisite section."""
        return self.sections[1]

    @prerequisite.setter
    def prerequisite(self, v):
        self.sections[1] = v

    @property
    def update(self) -> List[dns.rrset.RRset]:
        """The update section."""
        return self.sections[2]

    @update.setter
    def update(self, v):
        self.sections[2] = v

    def _add_rr(self, name, ttl, rd, deleting=None, section=None):
        """Add a single RR to the update section."""
        if section is None:
            section = self.update
        covers = rd.covers()
        rrset = self.find_rrset(section, name, self.zone_rdclass, rd.rdtype, covers, deleting, True, True)
        rrset.add(rd, ttl)

    def _add(self, replace, section, name, *args):
        """Add records.

        *replace* is the replacement mode.  If ``False``,
        RRs are added to an existing RRset; if ``True``, the RRset
        is replaced with the specified contents.  The second
        argument is the section to add to.  The third argument
        is always a name.  The other arguments can be:

                - rdataset...

                - ttl, rdata...

                - ttl, rdtype, string...
        """
        if isinstance(name, str):
            name = dns.name.from_text(name, None)
        if isinstance(args[0], dns.rdataset.Rdataset):
            for rds in args:
                if replace:
                    self.delete(name, rds.rdtype)
                for rd in rds:
                    self._add_rr(name, rds.ttl, rd, section=section)
        else:
            args = list(args)
            ttl = int(args.pop(0))
            if isinstance(args[0], dns.rdata.Rdata):
                if replace:
                    self.delete(name, args[0].rdtype)
                for rd in args:
                    self._add_rr(name, ttl, rd, section=section)
            else:
                rdtype = dns.rdatatype.RdataType.make(args.pop(0))
                if replace:
                    self.delete(name, rdtype)
                for s in args:
                    rd = dns.rdata.from_text(self.zone_rdclass, rdtype, s, self.origin)
                    self._add_rr(name, ttl, rd, section=section)

    def add(self, name: Union[dns.name.Name, str], *args: Any) -> None:
        """Add records.

        The first argument is always a name.  The other
        arguments can be:

                - rdataset...

                - ttl, rdata...

                - ttl, rdtype, string...
        """
        self._add(False, self.update, name, *args)

    def delete(self, name: Union[dns.name.Name, str], *args: Any) -> None:
        """Delete records.

        The first argument is always a name.  The other
        arguments can be:

                - *empty*

                - rdataset...

                - rdata...

                - rdtype, [string...]
        """
        if isinstance(name, str):
            name = dns.name.from_text(name, None)
        if len(args) == 0:
            self.find_rrset(self.update, name, dns.rdataclass.ANY, dns.rdatatype.ANY, dns.rdatatype.NONE, dns.rdataclass.ANY, True, True)
        elif isinstance(args[0], dns.rdataset.Rdataset):
            for rds in args:
                for rd in rds:
                    self._add_rr(name, 0, rd, dns.rdataclass.NONE)
        else:
            largs = list(args)
            if isinstance(largs[0], dns.rdata.Rdata):
                for rd in largs:
                    self._add_rr(name, 0, rd, dns.rdataclass.NONE)
            else:
                rdtype = dns.rdatatype.RdataType.make(largs.pop(0))
                if len(largs) == 0:
                    self.find_rrset(self.update, name, self.zone_rdclass, rdtype, dns.rdatatype.NONE, dns.rdataclass.ANY, True, True)
                else:
                    for s in largs:
                        rd = dns.rdata.from_text(self.zone_rdclass, rdtype, s, self.origin)
                        self._add_rr(name, 0, rd, dns.rdataclass.NONE)

    def replace(self, name: Union[dns.name.Name, str], *args: Any) -> None:
        """Replace records.

        The first argument is always a name.  The other
        arguments can be:

                - rdataset...

                - ttl, rdata...

                - ttl, rdtype, string...

        Note that if you want to replace the entire node, you should do
        a delete of the name followed by one or more calls to add.
        """
        self._add(True, self.update, name, *args)

    def present(self, name: Union[dns.name.Name, str], *args: Any) -> None:
        """Require that an owner name (and optionally an rdata type,
        or specific rdataset) exists as a prerequisite to the
        execution of the update.

        The first argument is always a name.
        The other arguments can be:

                - rdataset...

                - rdata...

                - rdtype, string...
        """
        if isinstance(name, str):
            name = dns.name.from_text(name, None)
        if len(args) == 0:
            self.find_rrset(self.prerequisite, name, dns.rdataclass.ANY, dns.rdatatype.ANY, dns.rdatatype.NONE, None, True, True)
        elif isinstance(args[0], dns.rdataset.Rdataset) or isinstance(args[0], dns.rdata.Rdata) or len(args) > 1:
            if not isinstance(args[0], dns.rdataset.Rdataset):
                largs = list(args)
                largs.insert(0, 0)
                self._add(False, self.prerequisite, name, *largs)
            else:
                self._add(False, self.prerequisite, name, *args)
        else:
            rdtype = dns.rdatatype.RdataType.make(args[0])
            self.find_rrset(self.prerequisite, name, dns.rdataclass.ANY, rdtype, dns.rdatatype.NONE, None, True, True)

    def absent(self, name: Union[dns.name.Name, str], rdtype: Optional[Union[dns.rdatatype.RdataType, str]]=None) -> None:
        """Require that an owner name (and optionally an rdata type) does
        not exist as a prerequisite to the execution of the update."""
        if isinstance(name, str):
            name = dns.name.from_text(name, None)
        if rdtype is None:
            self.find_rrset(self.prerequisite, name, dns.rdataclass.NONE, dns.rdatatype.ANY, dns.rdatatype.NONE, None, True, True)
        else:
            rdtype = dns.rdatatype.RdataType.make(rdtype)
            self.find_rrset(self.prerequisite, name, dns.rdataclass.NONE, rdtype, dns.rdatatype.NONE, None, True, True)

    def _get_one_rr_per_rrset(self, value):
        return True

    def _parse_rr_header(self, section, name, rdclass, rdtype):
        deleting = None
        empty = False
        if section == UpdateSection.ZONE:
            if dns.rdataclass.is_metaclass(rdclass) or rdtype != dns.rdatatype.SOA or self.zone:
                raise dns.exception.FormError
        else:
            if not self.zone:
                raise dns.exception.FormError
            if rdclass in (dns.rdataclass.ANY, dns.rdataclass.NONE):
                deleting = rdclass
                rdclass = self.zone[0].rdclass
                empty = deleting == dns.rdataclass.ANY or section == UpdateSection.PREREQ
        return (rdclass, rdtype, deleting, empty)