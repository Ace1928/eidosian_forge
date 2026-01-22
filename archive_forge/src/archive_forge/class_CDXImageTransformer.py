class CDXImageTransformer(object):

    def __init__(self, smiCol, width=1, verbose=1, tempHandler=None):
        self.smiCol = smiCol
        if tempHandler is None:
            tempHandler = ReportUtils.TempFileHandler()
        self.tempHandler = tempHandler
        self.width = width * inch
        self.verbose = verbose

    def __call__(self, arg):
        res = list(arg)
        if self.verbose:
            print('Render:', res[0])
        if hasCDX:
            smi = res[self.smiCol]
            tmpName = self.tempHandler.get('.jpg')
            try:
                img = chemdraw.SmilesToPilImage(smi)
                w, h = img.size
                aspect = float(h) / w
                img.save(tmpName)
                img = platypus.Image(tmpName)
                img.drawWidth = self.width
                img.drawHeight = aspect * self.width
                res[self.smiCol] = img
            except Exception:
                import traceback
                traceback.print_exc()
                res[self.smiCol] = 'Failed'
        return res