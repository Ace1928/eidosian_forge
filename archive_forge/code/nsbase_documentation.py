from io import BytesIO
import dns.exception
import dns.rdata
import dns.name
Base class for rdata that is like an NS record, but whose name
    is not compressed when convert to DNS wire format, and whose
    digestable form is not downcased.