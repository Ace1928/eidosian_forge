import re
from collections import defaultdict
from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import ElementTree, XMLCorpusReader
from nltk.util import LazyConcatenation, LazyMap, flatten
def webview_file(self, fileid, urlbase=None):
    """Map a corpus file to its web version on the CHILDES website,
        and open it in a web browser.

        The complete URL to be used is:
            childes.childes_url_base + urlbase + fileid.replace('.xml', '.cha')

        If no urlbase is passed, we try to calculate it.  This
        requires that the childes corpus was set up to mirror the
        folder hierarchy under childes.psy.cmu.edu/data-xml/, e.g.:
        nltk_data/corpora/childes/Eng-USA/Cornell/??? or
        nltk_data/corpora/childes/Romance/Spanish/Aguirre/???

        The function first looks (as a special case) if "Eng-USA" is
        on the path consisting of <corpus root>+fileid; then if
        "childes", possibly followed by "data-xml", appears. If neither
        one is found, we use the unmodified fileid and hope for the best.
        If this is not right, specify urlbase explicitly, e.g., if the
        corpus root points to the Cornell folder, urlbase='Eng-USA/Cornell'.
        """
    import webbrowser
    if urlbase:
        path = urlbase + '/' + fileid
    else:
        full = self.root + '/' + fileid
        full = re.sub('\\\\', '/', full)
        if '/childes/' in full.lower():
            path = re.findall('(?i)/childes(?:/data-xml)?/(.*)\\.xml', full)[0]
        elif 'eng-usa' in full.lower():
            path = 'Eng-USA/' + re.findall('/(?i)Eng-USA/(.*)\\.xml', full)[0]
        else:
            path = fileid
    if path.endswith('.xml'):
        path = path[:-4]
    if not path.endswith('.cha'):
        path = path + '.cha'
    url = self.childes_url_base + path
    webbrowser.open_new_tab(url)
    print('Opening in browser:', url)