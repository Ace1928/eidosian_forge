import html
import json
import pickle
import re
import warnings
from importlib import import_module
from os import path
from typing import IO, Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from docutils import nodes
from docutils.nodes import Element, Node
from sphinx import addnodes, package_dir
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.environment import BuildEnvironment
from sphinx.util import split_into
from sphinx.search.en import SearchEnglish
class IndexBuilder:
    """
    Helper class that creates a search index based on the doctrees
    passed to the `feed` method.
    """
    formats = {'json': json, 'pickle': pickle}

    def __init__(self, env: BuildEnvironment, lang: str, options: Dict, scoring: str) -> None:
        self.env = env
        self._titles: Dict[str, str] = {}
        self._filenames: Dict[str, str] = {}
        self._mapping: Dict[str, Set[str]] = {}
        self._title_mapping: Dict[str, Set[str]] = {}
        self._all_titles: Dict[str, List[Tuple[str, str]]] = {}
        self._index_entries: Dict[str, List[Tuple[str, str, str]]] = {}
        self._stem_cache: Dict[str, str] = {}
        self._objtypes: Dict[Tuple[str, str], int] = {}
        self._objnames: Dict[int, Tuple[str, str, str]] = {}
        lang_class = languages.get(lang)
        if lang_class is None and '_' in lang:
            lang_class = languages.get(lang.split('_')[0])
        if lang_class is None:
            self.lang: SearchLanguage = SearchEnglish(options)
        elif isinstance(lang_class, str):
            module, classname = lang_class.rsplit('.', 1)
            lang_class: Type[SearchLanguage] = getattr(import_module(module), classname)
            self.lang = lang_class(options)
        else:
            self.lang = lang_class(options)
        if scoring:
            with open(scoring, 'rb') as fp:
                self.js_scorer_code = fp.read().decode()
        else:
            self.js_scorer_code = ''
        self.js_splitter_code = ''

    def load(self, stream: IO, format: Any) -> None:
        """Reconstruct from frozen data."""
        if format == 'jsdump':
            warnings.warn('format=jsdump is deprecated, use json instead', RemovedInSphinx70Warning, stacklevel=2)
            format = self.formats['json']
        elif isinstance(format, str):
            format = self.formats[format]
        frozen = format.load(stream)
        if not isinstance(frozen, dict) or frozen.get('envversion') != self.env.version:
            raise ValueError('old format')
        index2fn = frozen['docnames']
        self._filenames = dict(zip(index2fn, frozen['filenames']))
        self._titles = dict(zip(index2fn, frozen['titles']))
        self._all_titles = {}
        for docname in self._titles.keys():
            self._all_titles[docname] = []
        for title, doc_tuples in frozen['alltitles'].items():
            for doc, titleid in doc_tuples:
                self._all_titles[index2fn[doc]].append((title, titleid))

        def load_terms(mapping: Dict[str, Any]) -> Dict[str, Set[str]]:
            rv = {}
            for k, v in mapping.items():
                if isinstance(v, int):
                    rv[k] = {index2fn[v]}
                else:
                    rv[k] = {index2fn[i] for i in v}
            return rv
        self._mapping = load_terms(frozen['terms'])
        self._title_mapping = load_terms(frozen['titleterms'])

    def dump(self, stream: IO, format: Any) -> None:
        """Dump the frozen index to a stream."""
        if format == 'jsdump':
            warnings.warn('format=jsdump is deprecated, use json instead', RemovedInSphinx70Warning, stacklevel=2)
            format = self.formats['json']
        elif isinstance(format, str):
            format = self.formats[format]
        format.dump(self.freeze(), stream)

    def get_objects(self, fn2index: Dict[str, int]) -> Dict[str, List[Tuple[int, int, int, str, str]]]:
        rv: Dict[str, List[Tuple[int, int, int, str, str]]] = {}
        otypes = self._objtypes
        onames = self._objnames
        for domainname, domain in sorted(self.env.domains.items()):
            for fullname, dispname, type, docname, anchor, prio in sorted(domain.get_objects()):
                if docname not in fn2index:
                    continue
                if prio < 0:
                    continue
                fullname = html.escape(fullname)
                dispname = html.escape(dispname)
                prefix, _, name = dispname.rpartition('.')
                plist = rv.setdefault(prefix, [])
                try:
                    typeindex = otypes[domainname, type]
                except KeyError:
                    typeindex = len(otypes)
                    otypes[domainname, type] = typeindex
                    otype = domain.object_types.get(type)
                    if otype:
                        onames[typeindex] = (domainname, type, str(domain.get_type_name(otype)))
                    else:
                        onames[typeindex] = (domainname, type, type)
                if anchor == fullname:
                    shortanchor = ''
                elif anchor == type + '-' + fullname:
                    shortanchor = '-'
                else:
                    shortanchor = anchor
                plist.append((fn2index[docname], typeindex, prio, shortanchor, name))
        return rv

    def get_terms(self, fn2index: Dict) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        rvs: Tuple[Dict[str, List[str]], Dict[str, List[str]]] = ({}, {})
        for rv, mapping in zip(rvs, (self._mapping, self._title_mapping)):
            for k, v in mapping.items():
                if len(v) == 1:
                    fn, = v
                    if fn in fn2index:
                        rv[k] = fn2index[fn]
                else:
                    rv[k] = sorted([fn2index[fn] for fn in v if fn in fn2index])
        return rvs

    def freeze(self) -> Dict[str, Any]:
        """Create a usable data structure for serializing."""
        docnames, titles = zip(*sorted(self._titles.items()))
        filenames = [self._filenames.get(docname) for docname in docnames]
        fn2index = {f: i for i, f in enumerate(docnames)}
        terms, title_terms = self.get_terms(fn2index)
        objects = self.get_objects(fn2index)
        objtypes = {v: k[0] + ':' + k[1] for k, v in self._objtypes.items()}
        objnames = self._objnames
        alltitles: Dict[str, List[Tuple[int, str]]] = {}
        for docname, titlelist in self._all_titles.items():
            for title, titleid in titlelist:
                alltitles.setdefault(title, []).append((fn2index[docname], titleid))
        index_entries: Dict[str, List[Tuple[int, str]]] = {}
        for docname, entries in self._index_entries.items():
            for entry, entry_id, main_entry in entries:
                index_entries.setdefault(entry.lower(), []).append((fn2index[docname], entry_id))
        return dict(docnames=docnames, filenames=filenames, titles=titles, terms=terms, objects=objects, objtypes=objtypes, objnames=objnames, titleterms=title_terms, envversion=self.env.version, alltitles=alltitles, indexentries=index_entries)

    def label(self) -> str:
        return '%s (code: %s)' % (self.lang.language_name, self.lang.lang)

    def prune(self, docnames: Iterable[str]) -> None:
        """Remove data for all docnames not in the list."""
        new_titles = {}
        new_alltitles = {}
        new_filenames = {}
        for docname in docnames:
            if docname in self._titles:
                new_titles[docname] = self._titles[docname]
                new_alltitles[docname] = self._all_titles[docname]
                new_filenames[docname] = self._filenames[docname]
        self._titles = new_titles
        self._filenames = new_filenames
        self._all_titles = new_alltitles
        for wordnames in self._mapping.values():
            wordnames.intersection_update(docnames)
        for wordnames in self._title_mapping.values():
            wordnames.intersection_update(docnames)

    def feed(self, docname: str, filename: str, title: str, doctree: nodes.document) -> None:
        """Feed a doctree to the index."""
        self._titles[docname] = title
        self._filenames[docname] = filename
        visitor = WordCollector(doctree, self.lang)
        doctree.walk(visitor)

        def stem(word: str) -> str:
            try:
                return self._stem_cache[word]
            except KeyError:
                self._stem_cache[word] = self.lang.stem(word).lower()
                return self._stem_cache[word]
        _filter = self.lang.word_filter
        self._all_titles[docname] = visitor.found_titles
        for word in visitor.found_title_words:
            stemmed_word = stem(word)
            if _filter(stemmed_word):
                self._title_mapping.setdefault(stemmed_word, set()).add(docname)
            elif _filter(word):
                self._title_mapping.setdefault(word, set()).add(docname)
        for word in visitor.found_words:
            stemmed_word = stem(word)
            if not _filter(stemmed_word) and _filter(word):
                stemmed_word = word
            already_indexed = docname in self._title_mapping.get(stemmed_word, set())
            if _filter(stemmed_word) and (not already_indexed):
                self._mapping.setdefault(stemmed_word, set()).add(docname)
        _index_entries: Set[Tuple[str, str, str]] = set()
        for node in doctree.findall(addnodes.index):
            for entry_type, value, tid, main, *index_key in node['entries']:
                tid = tid or ''
                try:
                    if entry_type == 'single':
                        try:
                            entry, subentry = split_into(2, 'single', value)
                        except ValueError:
                            entry, = split_into(1, 'single', value)
                            subentry = ''
                        _index_entries.add((entry, tid, main))
                        if subentry:
                            _index_entries.add((subentry, tid, main))
                    elif entry_type == 'pair':
                        first, second = split_into(2, 'pair', value)
                        _index_entries.add((first, tid, main))
                        _index_entries.add((second, tid, main))
                    elif entry_type == 'triple':
                        first, second, third = split_into(3, 'triple', value)
                        _index_entries.add((first, tid, main))
                        _index_entries.add((second, tid, main))
                        _index_entries.add((third, tid, main))
                    elif entry_type in {'see', 'seealso'}:
                        first, second = split_into(2, 'see', value)
                        _index_entries.add((first, tid, main))
                except ValueError:
                    pass
        self._index_entries[docname] = sorted(_index_entries)

    def context_for_searchtool(self) -> Dict[str, Any]:
        if self.lang.js_splitter_code:
            js_splitter_code = self.lang.js_splitter_code
        else:
            js_splitter_code = self.js_splitter_code
        return {'search_language_stemming_code': self.get_js_stemmer_code(), 'search_language_stop_words': json.dumps(sorted(self.lang.stopwords)), 'search_scorer_tool': self.js_scorer_code, 'search_word_splitter_code': js_splitter_code}

    def get_js_stemmer_rawcodes(self) -> List[str]:
        """Returns a list of non-minified stemmer JS files to copy."""
        if self.lang.js_stemmer_rawcode:
            return [path.join(package_dir, 'search', 'non-minified-js', fname) for fname in ('base-stemmer.js', self.lang.js_stemmer_rawcode)]
        else:
            return []

    def get_js_stemmer_rawcode(self) -> Optional[str]:
        return None

    def get_js_stemmer_code(self) -> str:
        """Returns JS code that will be inserted into language_data.js."""
        if self.lang.js_stemmer_rawcode:
            js_dir = path.join(package_dir, 'search', 'minified-js')
            with open(path.join(js_dir, 'base-stemmer.js'), encoding='utf-8') as js_file:
                base_js = js_file.read()
            with open(path.join(js_dir, self.lang.js_stemmer_rawcode), encoding='utf-8') as js_file:
                language_js = js_file.read()
            return '%s\n%s\nStemmer = %sStemmer;' % (base_js, language_js, self.lang.language_name)
        else:
            return self.lang.js_stemmer_code