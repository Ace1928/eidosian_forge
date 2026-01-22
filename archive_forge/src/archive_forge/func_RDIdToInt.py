from rdkit import RDConfig
from rdkit.Dbase import DbModule
def RDIdToInt(ID, validate=1):
    """ Returns the integer index for a given RDId
  Throws a ValueError on error

  >>> RDIdToInt('RDCmpd-000-009-9')
  9
  >>> RDIdToInt('RDCmpd-009-000-009-8')
  9000009
  >>> RDIdToInt('RDData_000_009_9')
  9
  >>> try:
  ...   RDIdToInt('RDCmpd-009-000-109-8')
  ... except ValueError:
  ...   print('ok')
  ... else:
  ...   print('failed')
  ok
  >>> try:
  ...   RDIdToInt('bogus')
  ... except ValueError:
  ...   print('ok')
  ... else:
  ...   print('failed')
  ok

  """
    if validate and (not ValidateRDId(ID)):
        raise ValueError('Bad RD Id')
    ID = ID.replace('_', '-')
    terms = ID.split('-')[1:-1]
    res = 0
    factor = 1
    terms.reverse()
    for term in terms:
        res += factor * int(term)
        factor *= 1000
    return res