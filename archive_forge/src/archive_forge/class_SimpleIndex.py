from reportlab.lib.units import cm
from reportlab.lib.utils import commasplit, escapeOnce, encode_label, decode_label, strTypes, asUnicode, asNative
from reportlab.lib.styles import ParagraphStyle, _baseFontName
from reportlab.lib import sequencer as rl_sequencer
from reportlab.platypus.paragraph import Paragraph
from reportlab.platypus.doctemplate import IndexingFlowable
from reportlab.platypus.tables import TableStyle, Table
from reportlab.platypus.flowables import Spacer
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
import unicodedata
from ast import literal_eval
class SimpleIndex(IndexingFlowable):
    """Creates multi level indexes.
    The styling can be cutomized and alphabetic headers turned on and off.
    """

    def __init__(self, **kwargs):
        """
        Constructor of SimpleIndex.
        Accepts the same arguments as the setup method.
        """
        self._entries = {}
        self._lastEntries = {}
        self._flowable = None
        self.setup(**kwargs)

    def getFormatFunc(self, formatName):
        try:
            return getattr(rl_sequencer, '_format_%s' % formatName)
        except ImportError:
            raise ValueError('Unknown sequencer format %r' % formatName)

    def setup(self, style=None, dot=None, tableStyle=None, headers=True, name=None, format='123', offset=0):
        """
        This method makes it possible to change styling and other parameters on an existing object.

        style is the paragraph style to use for index entries.
        dot can either be None or a string. If it's None, entries are immediatly followed by their
            corresponding page numbers. If it's a string, page numbers are aligned on the right side
            of the document and the gap filled with a repeating sequence of the string.
        tableStyle is the style used by the table which the index uses to draw itself. Use this to
            change properties like spacing between elements.
        headers is a boolean. If it is True, alphabetic headers are displayed in the Index when the first
        letter changes. If False, we just output some extra space before the next item
        name makes it possible to use several indexes in one document. If you want this use this
            parameter to give each index a unique name. You can then index a term by refering to the
            name of the index which it should appear in:

                <index item="term" name="myindex" />

        format can be 'I', 'i', '123',  'ABC', 'abc'
        """
        if style is None:
            style = ParagraphStyle(name='index', fontName=_baseFontName, fontSize=11)
        self.textStyle = style
        self.tableStyle = tableStyle or defaultTableStyle
        self.dot = dot
        self.headers = headers
        if name is None:
            from reportlab.platypus.paraparser import DEFAULT_INDEX_NAME as name
        self.name = name
        self.formatFunc = self.getFormatFunc(format)
        self.offset = offset

    def __call__(self, canv, kind, label):
        label = asNative(label, 'latin1')
        try:
            terms, format, offset = decode_label(label)
        except:
            terms = label
            format = offset = None
        if format is None:
            formatFunc = self.formatFunc
        else:
            formatFunc = self.getFormatFunc(format)
        if offset is None:
            offset = self.offset
        terms = commasplit(terms)
        cPN = canv.getPageNumber()
        pns = formatFunc(cPN - offset)
        key = 'ix_%s_%s_p_%s' % (self.name, label, pns)
        info = canv._curr_tx_info
        canv.bookmarkHorizontal(key, info['cur_x'], info['cur_y'] + info['leading'])
        self.addEntry(terms, (cPN, pns), key)

    def getCanvasMaker(self, canvasmaker=canvas.Canvas):

        def newcanvasmaker(*args, **kwargs):
            from reportlab.pdfgen import canvas
            c = canvasmaker(*args, **kwargs)
            setattr(c, self.name, self)
            return c
        return newcanvasmaker

    def isIndexing(self):
        return 1

    def isSatisfied(self):
        return self._entries == self._lastEntries

    def beforeBuild(self):
        self._lastEntries = self._entries.copy()
        self.clearEntries()

    def clearEntries(self):
        self._entries = {}

    def notify(self, kind, stuff):
        """The notification hook called to register all kinds of events.

        Here we are interested in 'IndexEntry' events only.
        """
        if kind == 'IndexEntry':
            text, pageNum = stuff
            self.addEntry(text, (self._canv.getPageNumber(), pageNum))

    def addEntry(self, text, pageNum, key=None):
        """Allows incremental buildup"""
        self._entries.setdefault(makeTuple(text), set([])).add((pageNum, key))

    def split(self, availWidth, availHeight):
        """At this stage we do not care about splitting the entries,
        we will just return a list of platypus tables.  Presumably the
        calling app has a pointer to the original TableOfContents object;
        Platypus just sees tables.
        """
        return self._flowable.splitOn(self.canv, availWidth, availHeight)

    def _getlastEntries(self, dummy=[(['Placeholder for index'], enumerate((None,) * 3))]):
        """Return the last run's entries!  If there are none, returns dummy."""
        lE = self._lastEntries or self._entries
        if not lE:
            return dummy
        return list(sorted(lE.items()))

    def _build(self, availWidth, availHeight):
        _tempEntries = [(tuple((asUnicode(t) for t in texts)), pageNumbers) for texts, pageNumbers in self._getlastEntries()]

        def getkey(seq):
            return [''.join((c for c in unicodedata.normalize('NFD', x.upper()) if unicodedata.category(c) != 'Mn')) for x in seq[0]]
        _tempEntries.sort(key=getkey)
        leveloffset = self.headers and 1 or 0

        def drawIndexEntryEnd(canvas, kind, label):
            """Callback to draw dots and page numbers after each entry."""
            style = self.getLevelStyle(leveloffset)
            pages = [(p[1], k) for p, k in sorted(decode_label(label))]
            drawPageNumbers(canvas, style, pages, availWidth, availHeight, self.dot)
        self.canv.drawIndexEntryEnd = drawIndexEntryEnd
        alpha = ''
        tableData = []
        lastTexts = []
        alphaStyle = self.getLevelStyle(0)
        for texts, pageNumbers in _tempEntries:
            texts = list(texts)
            nalpha = ''.join((c for c in unicodedata.normalize('NFD', texts[0][0].upper()) if unicodedata.category(c) != 'Mn'))
            if alpha != nalpha:
                alpha = nalpha
                if self.headers:
                    header = alpha
                else:
                    header = ' '
                tableData.append([Spacer(1, alphaStyle.spaceBefore)])
                tableData.append([Paragraph(header, alphaStyle)])
                tableData.append([Spacer(1, alphaStyle.spaceAfter)])
            i, diff = listdiff(lastTexts, texts)
            if diff:
                lastTexts = texts
                texts = texts[i:]
            label = encode_label(list(pageNumbers))
            texts[-1] = '%s<onDraw name="drawIndexEntryEnd" label="%s"/>' % (texts[-1], label)
            for text in texts:
                text = escapeOnce(text)
                style = self.getLevelStyle(i + leveloffset)
                para = Paragraph(text, style)
                if style.spaceBefore:
                    tableData.append([Spacer(1, style.spaceBefore)])
                tableData.append([para])
                i += 1
        self._flowable = Table(tableData, colWidths=[availWidth], style=self.tableStyle)

    def wrap(self, availWidth, availHeight):
        """All table properties should be known by now."""
        self._build(availWidth, availHeight)
        self.width, self.height = self._flowable.wrapOn(self.canv, availWidth, availHeight)
        return (self.width, self.height)

    def drawOn(self, canvas, x, y, _sW=0):
        """Don't do this at home!  The standard calls for implementing
        draw(); we are hooking this in order to delegate ALL the drawing
        work to the embedded table object.
        """
        self._flowable.drawOn(canvas, x, y, _sW)

    def draw(self):
        t = self._flowable
        ocanv = getattr(t, 'canv', None)
        if not ocanv:
            t.canv = self.canv
        try:
            t.draw()
        finally:
            if not ocanv:
                del t.canv

    def getLevelStyle(self, n):
        """Returns the style for level n, generating and caching styles on demand if not present."""
        if not hasattr(self.textStyle, '__iter__'):
            self.textStyle = [self.textStyle]
        try:
            return self.textStyle[n]
        except IndexError:
            self.textStyle = list(self.textStyle)
            prevstyle = self.getLevelStyle(n - 1)
            self.textStyle.append(ParagraphStyle(name='%s-%d-indented' % (prevstyle.name, n), parent=prevstyle, firstLineIndent=prevstyle.firstLineIndent + 0.2 * cm, leftIndent=prevstyle.leftIndent + 0.2 * cm))
            return self.textStyle[n]