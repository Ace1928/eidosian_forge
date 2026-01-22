import struct
def _read_v4(f):
    record = Record()
    preHeaders = ['version', 'columns', 'rows', 'cellNo', 'headerLen']
    preHeadersMap = {}
    headersMap = {}
    preHeadersMap['magic'] = 64
    try:
        for name in preHeaders:
            preHeadersMap[name] = struct.unpack('<i', f.read(4))[0]
    except struct.error:
        raise ParserError('Failed to parse CEL version 4 file') from None
    char = f.read(preHeadersMap['headerLen'])
    header = char.decode('ascii', 'ignore')
    for line in header.split('\n'):
        if '=' in line:
            headline = line.split('=')
            headersMap[headline[0]] = '='.join(headline[1:])
    record.version = preHeadersMap['version']
    if record.version != 4:
        raise ParserError('Incorrect version number in CEL version 4 file')
    record.GridCornerUL = headersMap['GridCornerUL']
    record.GridCornerUR = headersMap['GridCornerUR']
    record.GridCornerLR = headersMap['GridCornerLR']
    record.GridCornerLL = headersMap['GridCornerLL']
    record.DatHeader = headersMap['DatHeader']
    record.Algorithm = headersMap['Algorithm']
    record.AlgorithmParameters = headersMap['AlgorithmParameters']
    record.NumberCells = preHeadersMap['cellNo']
    record.nrows = int(headersMap['Rows'])
    record.ncols = int(headersMap['Cols'])
    record.nmask = None
    record.mask = None
    record.noutliers = None
    record.outliers = None
    record.modified = None

    def raiseBadHeader(field, expected):
        actual = int(headersMap[field])
        message = f'The header {field} is expected to be 0, not {actual}'
        if actual != expected:
            raise ParserError(message)
    raiseBadHeader('Axis-invertX', 0)
    raiseBadHeader('AxisInvertY', 0)
    raiseBadHeader('OffsetX', 0)
    raiseBadHeader('OffsetY', 0)
    char = b'\x00'
    safetyValve = 10 ** 4
    for i in range(safetyValve):
        char = f.read(1)
        if char == b'\x04':
            break
        if i == safetyValve:
            raise ParserError('Parse Error. The parser expects a short, undocumented binary blob terminating with ASCII EOF, x04')
    padding = f.read(15)
    structa = struct.Struct('< f f h')
    structSize = 10
    record.intensities = np.empty(record.NumberCells, dtype=float)
    record.stdevs = np.empty(record.NumberCells, dtype=float)
    record.npix = np.empty(record.NumberCells, dtype=int)
    b = f.read(structSize * record.NumberCells)
    for i in range(record.NumberCells):
        binaryFragment = b[i * structSize:(i + 1) * structSize]
        intensity, stdevs, npix = structa.unpack(binaryFragment)
        record.intensities[i] = intensity
        record.stdevs[i] = stdevs
        record.npix[i] = npix

    def reshape(array):
        view = array.view()
        view.shape = (record.nrows, record.ncols)
        return view
    record.intensities = reshape(record.intensities)
    record.stdevs = reshape(record.stdevs)
    record.npix = reshape(record.npix)
    return record