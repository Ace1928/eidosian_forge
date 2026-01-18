from suds.sax.enc import Encoder
@classmethod
def xsd(cls, ns):
    try:
        return cls.w3(ns) and ns[1].endswith('XMLSchema')
    except Exception:
        pass
    return False