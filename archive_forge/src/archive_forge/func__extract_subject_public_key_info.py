from Cryptodome.Util.asn1 import (DerSequence, DerInteger, DerBitString,
def _extract_subject_public_key_info(x509_certificate):
    """Extract subjectPublicKeyInfo from a DER X.509 certificate."""
    certificate = DerSequence().decode(x509_certificate, nr_elements=3)
    tbs_certificate = DerSequence().decode(certificate[0], nr_elements=range(6, 11))
    index = 5
    try:
        tbs_certificate[0] + 1
        version = 1
    except TypeError:
        version = DerInteger(explicit=0).decode(tbs_certificate[0]).value
        if version not in (2, 3):
            raise ValueError('Incorrect X.509 certificate version')
        index = 6
    return tbs_certificate[index]