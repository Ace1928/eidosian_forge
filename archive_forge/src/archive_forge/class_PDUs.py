from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1_modules import rfc1902
class PDUs(univ.Choice):
    componentType = namedtype.NamedTypes(namedtype.NamedType('get-request', GetRequestPDU()), namedtype.NamedType('get-next-request', GetNextRequestPDU()), namedtype.NamedType('get-bulk-request', GetBulkRequestPDU()), namedtype.NamedType('response', ResponsePDU()), namedtype.NamedType('set-request', SetRequestPDU()), namedtype.NamedType('inform-request', InformRequestPDU()), namedtype.NamedType('snmpV2-trap', SNMPv2TrapPDU()), namedtype.NamedType('report', ReportPDU()))