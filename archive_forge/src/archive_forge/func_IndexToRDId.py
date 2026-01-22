from rdkit import RDConfig
from rdkit.Dbase import DbModule
def IndexToRDId(idx, leadText='RDCmpd'):
    """ Converts an integer index into an RDId

  The format of the ID is:
    leadText-xxx-xxx-xxx-y
  The number blocks are zero padded and the final digit (y)
  is a checksum:

  >>> str(IndexToRDId(9))
  'RDCmpd-000-009-9'
  >>> str(IndexToRDId(9009))
  'RDCmpd-009-009-8'

  A millions block is included if it's nonzero:

  >>> str(IndexToRDId(9000009))
  'RDCmpd-009-000-009-8'

  The text at the beginning can be altered:

  >>> str(IndexToRDId(9,leadText='RDAlt'))
  'RDAlt-000-009-9'

  Negative indices are errors:

  >>> try:
  ...   IndexToRDId(-1)
  ... except ValueError:
  ...   print('ok')
  ... else:
  ...   print('failed')
  ok

  """
    if idx < 0:
        raise ValueError('indices must be >= zero')
    res = leadText + '-'
    tmpIdx = idx
    if idx >= 1000000.0:
        res += '%03d-' % (idx // 1000000.0)
        tmpIdx = idx % int(1000000.0)
    if tmpIdx < 1000:
        res += '000-'
    else:
        res += '%03d-' % (tmpIdx // 1000)
        tmpIdx = tmpIdx % 1000
    res += '%03d-' % tmpIdx
    accum = 0
    txt = str(idx)
    for char in txt:
        accum += int(char)
    res += str(accum % 10)
    return res