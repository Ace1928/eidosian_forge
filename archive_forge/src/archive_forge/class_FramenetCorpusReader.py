import itertools
import os
import re
import sys
import textwrap
import types
from collections import OrderedDict, defaultdict
from itertools import zip_longest
from operator import itemgetter
from pprint import pprint
from nltk.corpus.reader import XMLCorpusReader, XMLCorpusView
from nltk.util import LazyConcatenation, LazyIteratorList, LazyMap
class FramenetCorpusReader(XMLCorpusReader):
    """A corpus reader for the Framenet Corpus.

    >>> from nltk.corpus import framenet as fn
    >>> fn.lu(3238).frame.lexUnit['glint.v'] is fn.lu(3238)
    True
    >>> fn.frame_by_name('Replacing') is fn.lus('replace.v')[0].frame
    True
    >>> fn.lus('prejudice.n')[0].frame.frameRelations == fn.frame_relations('Partiality')
    True
    """
    _bad_statuses = ['Problem']
    "\n    When loading LUs for a frame, those whose status is in this list will be ignored.\n    Due to caching, if user code modifies this, it should do so before loading any data.\n    'Problem' should always be listed for FrameNet 1.5, as these LUs are not included\n    in the XML index.\n    "
    _warnings = False

    def warnings(self, v):
        """Enable or disable warnings of data integrity issues as they are encountered.
        If v is truthy, warnings will be enabled.

        (This is a function rather than just an attribute/property to ensure that if
        enabling warnings is the first action taken, the corpus reader is instantiated first.)
        """
        self._warnings = v

    def __init__(self, root, fileids):
        XMLCorpusReader.__init__(self, root, fileids)
        self._frame_dir = 'frame'
        self._lu_dir = 'lu'
        self._fulltext_dir = 'fulltext'
        self._fnweb_url = 'https://framenet2.icsi.berkeley.edu/fnReports/data'
        self._frame_idx = None
        self._cached_frames = {}
        self._lu_idx = None
        self._fulltext_idx = None
        self._semtypes = None
        self._freltyp_idx = None
        self._frel_idx = None
        self._ferel_idx = None
        self._frel_f_idx = None
        self._readme = 'README.txt'

    def help(self, attrname=None):
        """Display help information summarizing the main methods."""
        if attrname is not None:
            return help(self.__getattribute__(attrname))
        msg = '\nCitation: Nathan Schneider and Chuck Wooters (2017),\n"The NLTK FrameNet API: Designing for Discoverability with a Rich Linguistic Resource".\nProceedings of EMNLP: System Demonstrations. https://arxiv.org/abs/1703.07438\n\nUse the following methods to access data in FrameNet.\nProvide a method name to `help()` for more information.\n\nFRAMES\n======\n\nframe() to look up a frame by its exact name or ID\nframes() to get frames matching a name pattern\nframes_by_lemma() to get frames containing an LU matching a name pattern\nframe_ids_and_names() to get a mapping from frame IDs to names\n\nFRAME ELEMENTS\n==============\n\nfes() to get frame elements (a.k.a. roles) matching a name pattern, optionally constrained\n  by a frame name pattern\n\nLEXICAL UNITS\n=============\n\nlu() to look up an LU by its ID\nlus() to get lexical units matching a name pattern, optionally constrained by frame\nlu_ids_and_names() to get a mapping from LU IDs to names\n\nRELATIONS\n=========\n\nframe_relation_types() to get the different kinds of frame-to-frame relations\n  (Inheritance, Subframe, Using, etc.).\nframe_relations() to get the relation instances, optionally constrained by\n  frame(s) or relation type\nfe_relations() to get the frame element pairs belonging to a frame-to-frame relation\n\nSEMANTIC TYPES\n==============\n\nsemtypes() to get the different kinds of semantic types that can be applied to\n  FEs, LUs, and entire frames\nsemtype() to look up a particular semtype by name, ID, or abbreviation\nsemtype_inherits() to check whether two semantic types have a subtype-supertype\n  relationship in the semtype hierarchy\npropagate_semtypes() to apply inference rules that distribute semtypes over relations\n  between FEs\n\nANNOTATIONS\n===========\n\nannotations() to get annotation sets, in which a token in a sentence is annotated\n  with a lexical unit in a frame, along with its frame elements and their syntactic properties;\n  can be constrained by LU name pattern and limited to lexicographic exemplars or full-text.\n  Sentences of full-text annotation can have multiple annotation sets.\nsents() to get annotated sentences illustrating one or more lexical units\nexemplars() to get sentences of lexicographic annotation, most of which have\n  just 1 annotation set; can be constrained by LU name pattern, frame, and overt FE(s)\ndoc() to look up a document of full-text annotation by its ID\ndocs() to get documents of full-text annotation that match a name pattern\ndocs_metadata() to get metadata about all full-text documents without loading them\nft_sents() to iterate over sentences of full-text annotation\n\nUTILITIES\n=========\n\nbuildindexes() loads metadata about all frames, LUs, etc. into memory to avoid\n  delay when one is accessed for the first time. It does not load annotations.\nreadme() gives the text of the FrameNet README file\nwarnings(True) to display corpus consistency warnings when loading data\n        '
        print(msg)

    def _buildframeindex(self):
        if not self._frel_idx:
            self._buildrelationindex()
        self._frame_idx = {}
        with XMLCorpusView(self.abspath('frameIndex.xml'), 'frameIndex/frame', self._handle_elt) as view:
            for f in view:
                self._frame_idx[f['ID']] = f

    def _buildcorpusindex(self):
        self._fulltext_idx = {}
        with XMLCorpusView(self.abspath('fulltextIndex.xml'), 'fulltextIndex/corpus', self._handle_fulltextindex_elt) as view:
            for doclist in view:
                for doc in doclist:
                    self._fulltext_idx[doc.ID] = doc

    def _buildluindex(self):
        self._lu_idx = {}
        with XMLCorpusView(self.abspath('luIndex.xml'), 'luIndex/lu', self._handle_elt) as view:
            for lu in view:
                self._lu_idx[lu['ID']] = lu

    def _buildrelationindex(self):
        self._freltyp_idx = {}
        self._frel_idx = {}
        self._frel_f_idx = defaultdict(set)
        self._ferel_idx = {}
        with XMLCorpusView(self.abspath('frRelation.xml'), 'frameRelations/frameRelationType', self._handle_framerelationtype_elt) as view:
            for freltyp in view:
                self._freltyp_idx[freltyp.ID] = freltyp
                for frel in freltyp.frameRelations:
                    supF = frel.superFrame = frel[freltyp.superFrameName] = Future((lambda fID: lambda: self.frame_by_id(fID))(frel.supID))
                    subF = frel.subFrame = frel[freltyp.subFrameName] = Future((lambda fID: lambda: self.frame_by_id(fID))(frel.subID))
                    self._frel_idx[frel.ID] = frel
                    self._frel_f_idx[frel.supID].add(frel.ID)
                    self._frel_f_idx[frel.subID].add(frel.ID)
                    for ferel in frel.feRelations:
                        ferel.superFrame = supF
                        ferel.subFrame = subF
                        ferel.superFE = Future((lambda fer: lambda: fer.superFrame.FE[fer.superFEName])(ferel))
                        ferel.subFE = Future((lambda fer: lambda: fer.subFrame.FE[fer.subFEName])(ferel))
                        self._ferel_idx[ferel.ID] = ferel

    def _warn(self, *message, **kwargs):
        if self._warnings:
            kwargs.setdefault('file', sys.stderr)
            print(*message, **kwargs)

    def buildindexes(self):
        """
        Build the internal indexes to make look-ups faster.
        """
        self._buildframeindex()
        self._buildluindex()
        self._buildcorpusindex()
        self._buildrelationindex()

    def doc(self, fn_docid):
        """
        Returns the annotated document whose id number is
        ``fn_docid``. This id number can be obtained by calling the
        Documents() function.

        The dict that is returned from this function will contain the
        following keys:

        - '_type'      : 'fulltextannotation'
        - 'sentence'   : a list of sentences in the document
           - Each item in the list is a dict containing the following keys:
              - 'ID'    : the ID number of the sentence
              - '_type' : 'sentence'
              - 'text'  : the text of the sentence
              - 'paragNo' : the paragraph number
              - 'sentNo'  : the sentence number
              - 'docID'   : the document ID number
              - 'corpID'  : the corpus ID number
              - 'aPos'    : the annotation position
              - 'annotationSet' : a list of annotation layers for the sentence
                 - Each item in the list is a dict containing the following keys:
                    - 'ID'       : the ID number of the annotation set
                    - '_type'    : 'annotationset'
                    - 'status'   : either 'MANUAL' or 'UNANN'
                    - 'luName'   : (only if status is 'MANUAL')
                    - 'luID'     : (only if status is 'MANUAL')
                    - 'frameID'  : (only if status is 'MANUAL')
                    - 'frameName': (only if status is 'MANUAL')
                    - 'layer' : a list of labels for the layer
                       - Each item in the layer is a dict containing the following keys:
                          - '_type': 'layer'
                          - 'rank'
                          - 'name'
                          - 'label' : a list of labels in the layer
                             - Each item is a dict containing the following keys:
                                - 'start'
                                - 'end'
                                - 'name'
                                - 'feID' (optional)

        :param fn_docid: The Framenet id number of the document
        :type fn_docid: int
        :return: Information about the annotated document
        :rtype: dict
        """
        try:
            xmlfname = self._fulltext_idx[fn_docid].filename
        except TypeError:
            self._buildcorpusindex()
            xmlfname = self._fulltext_idx[fn_docid].filename
        except KeyError as e:
            raise FramenetError(f'Unknown document id: {fn_docid}') from e
        locpath = os.path.join(f'{self._root}', self._fulltext_dir, xmlfname)
        with XMLCorpusView(locpath, 'fullTextAnnotation') as view:
            elt = view[0]
        info = self._handle_fulltextannotation_elt(elt)
        for k, v in self._fulltext_idx[fn_docid].items():
            info[k] = v
        return info

    def frame_by_id(self, fn_fid, ignorekeys=[]):
        """
        Get the details for the specified Frame using the frame's id
        number.

        Usage examples:

        >>> from nltk.corpus import framenet as fn
        >>> f = fn.frame_by_id(256)
        >>> f.ID
        256
        >>> f.name
        'Medical_specialties'
        >>> f.definition # doctest: +NORMALIZE_WHITESPACE
        "This frame includes words that name medical specialties and is closely related to the
        Medical_professionals frame.  The FE Type characterizing a sub-are in a Specialty may also be
        expressed. 'Ralph practices paediatric oncology.'"

        :param fn_fid: The Framenet id number of the frame
        :type fn_fid: int
        :param ignorekeys: The keys to ignore. These keys will not be
            included in the output. (optional)
        :type ignorekeys: list(str)
        :return: Information about a frame
        :rtype: dict

        Also see the ``frame()`` function for details about what is
        contained in the dict that is returned.
        """
        try:
            fentry = self._frame_idx[fn_fid]
            if '_type' in fentry:
                return fentry
            name = fentry['name']
        except TypeError:
            self._buildframeindex()
            name = self._frame_idx[fn_fid]['name']
        except KeyError as e:
            raise FramenetError(f'Unknown frame id: {fn_fid}') from e
        return self.frame_by_name(name, ignorekeys, check_cache=False)

    def frame_by_name(self, fn_fname, ignorekeys=[], check_cache=True):
        """
        Get the details for the specified Frame using the frame's name.

        Usage examples:

        >>> from nltk.corpus import framenet as fn
        >>> f = fn.frame_by_name('Medical_specialties')
        >>> f.ID
        256
        >>> f.name
        'Medical_specialties'
        >>> f.definition # doctest: +NORMALIZE_WHITESPACE
         "This frame includes words that name medical specialties and is closely related to the
          Medical_professionals frame.  The FE Type characterizing a sub-are in a Specialty may also be
          expressed. 'Ralph practices paediatric oncology.'"

        :param fn_fname: The name of the frame
        :type fn_fname: str
        :param ignorekeys: The keys to ignore. These keys will not be
            included in the output. (optional)
        :type ignorekeys: list(str)
        :return: Information about a frame
        :rtype: dict

        Also see the ``frame()`` function for details about what is
        contained in the dict that is returned.
        """
        if check_cache and fn_fname in self._cached_frames:
            return self._frame_idx[self._cached_frames[fn_fname]]
        elif not self._frame_idx:
            self._buildframeindex()
        locpath = os.path.join(f'{self._root}', self._frame_dir, fn_fname + '.xml')
        try:
            with XMLCorpusView(locpath, 'frame') as view:
                elt = view[0]
        except OSError as e:
            raise FramenetError(f'Unknown frame: {fn_fname}') from e
        fentry = self._handle_frame_elt(elt, ignorekeys)
        assert fentry
        fentry.URL = self._fnweb_url + '/' + self._frame_dir + '/' + fn_fname + '.xml'
        for st in fentry.semTypes:
            if st.rootType.name == 'Lexical_type':
                for lu in fentry.lexUnit.values():
                    if not any((x is st for x in lu.semTypes)):
                        lu.semTypes.append(st)
        self._frame_idx[fentry.ID] = fentry
        self._cached_frames[fentry.name] = fentry.ID
        '\n        # now set up callables to resolve the LU pointers lazily.\n        # (could also do this here--caching avoids infinite recursion.)\n        for luName,luinfo in fentry.lexUnit.items():\n            fentry.lexUnit[luName] = (lambda luID: Future(lambda: self.lu(luID)))(luinfo.ID)\n        '
        return fentry

    def frame(self, fn_fid_or_fname, ignorekeys=[]):
        """
        Get the details for the specified Frame using the frame's name
        or id number.

        Usage examples:

        >>> from nltk.corpus import framenet as fn
        >>> f = fn.frame(256)
        >>> f.name
        'Medical_specialties'
        >>> f = fn.frame('Medical_specialties')
        >>> f.ID
        256
        >>> # ensure non-ASCII character in definition doesn't trigger an encoding error:
        >>> fn.frame('Imposing_obligation') # doctest: +ELLIPSIS
        frame (1494): Imposing_obligation...


        The dict that is returned from this function will contain the
        following information about the Frame:

        - 'name'       : the name of the Frame (e.g. 'Birth', 'Apply_heat', etc.)
        - 'definition' : textual definition of the Frame
        - 'ID'         : the internal ID number of the Frame
        - 'semTypes'   : a list of semantic types for this frame
           - Each item in the list is a dict containing the following keys:
              - 'name' : can be used with the semtype() function
              - 'ID'   : can be used with the semtype() function

        - 'lexUnit'    : a dict containing all of the LUs for this frame.
                         The keys in this dict are the names of the LUs and
                         the value for each key is itself a dict containing
                         info about the LU (see the lu() function for more info.)

        - 'FE' : a dict containing the Frame Elements that are part of this frame
                 The keys in this dict are the names of the FEs (e.g. 'Body_system')
                 and the values are dicts containing the following keys

              - 'definition' : The definition of the FE
              - 'name'       : The name of the FE e.g. 'Body_system'
              - 'ID'         : The id number
              - '_type'      : 'fe'
              - 'abbrev'     : Abbreviation e.g. 'bod'
              - 'coreType'   : one of "Core", "Peripheral", or "Extra-Thematic"
              - 'semType'    : if not None, a dict with the following two keys:
                 - 'name' : name of the semantic type. can be used with
                            the semtype() function
                 - 'ID'   : id number of the semantic type. can be used with
                            the semtype() function
              - 'requiresFE' : if not None, a dict with the following two keys:
                 - 'name' : the name of another FE in this frame
                 - 'ID'   : the id of the other FE in this frame
              - 'excludesFE' : if not None, a dict with the following two keys:
                 - 'name' : the name of another FE in this frame
                 - 'ID'   : the id of the other FE in this frame

        - 'frameRelation'      : a list of objects describing frame relations
        - 'FEcoreSets'  : a list of Frame Element core sets for this frame
           - Each item in the list is a list of FE objects

        :param fn_fid_or_fname: The Framenet name or id number of the frame
        :type fn_fid_or_fname: int or str
        :param ignorekeys: The keys to ignore. These keys will not be
            included in the output. (optional)
        :type ignorekeys: list(str)
        :return: Information about a frame
        :rtype: dict
        """
        if isinstance(fn_fid_or_fname, str):
            f = self.frame_by_name(fn_fid_or_fname, ignorekeys)
        else:
            f = self.frame_by_id(fn_fid_or_fname, ignorekeys)
        return f

    def frames_by_lemma(self, pat):
        """
        Returns a list of all frames that contain LUs in which the
        ``name`` attribute of the LU matches the given regular expression
        ``pat``. Note that LU names are composed of "lemma.POS", where
        the "lemma" part can be made up of either a single lexeme
        (e.g. 'run') or multiple lexemes (e.g. 'a little').

        Note: if you are going to be doing a lot of this type of
        searching, you'd want to build an index that maps from lemmas to
        frames because each time frames_by_lemma() is called, it has to
        search through ALL of the frame XML files in the db.

        >>> from nltk.corpus import framenet as fn
        >>> from nltk.corpus.reader.framenet import PrettyList
        >>> PrettyList(sorted(fn.frames_by_lemma(r'(?i)a little'), key=itemgetter('ID'))) # doctest: +ELLIPSIS
        [<frame ID=189 name=Quanti...>, <frame ID=2001 name=Degree>]

        :return: A list of frame objects.
        :rtype: list(AttrDict)
        """
        return PrettyList((f for f in self.frames() if any((re.search(pat, luName) for luName in f.lexUnit))))

    def lu_basic(self, fn_luid):
        """
        Returns basic information about the LU whose id is
        ``fn_luid``. This is basically just a wrapper around the
        ``lu()`` function with "subCorpus" info excluded.

        >>> from nltk.corpus import framenet as fn
        >>> lu = PrettyDict(fn.lu_basic(256), breakLines=True)
        >>> # ellipses account for differences between FN 1.5 and 1.7
        >>> lu # doctest: +ELLIPSIS
        {'ID': 256,
         'POS': 'V',
         'URL': 'https://framenet2.icsi.berkeley.edu/fnReports/data/lu/lu256.xml',
         '_type': 'lu',
         'cBy': ...,
         'cDate': '02/08/2001 01:27:50 PST Thu',
         'definition': 'COD: be aware of beforehand; predict.',
         'definitionMarkup': 'COD: be aware of beforehand; predict.',
         'frame': <frame ID=26 name=Expectation>,
         'lemmaID': 15082,
         'lexemes': [{'POS': 'V', 'breakBefore': 'false', 'headword': 'false', 'name': 'foresee', 'order': 1}],
         'name': 'foresee.v',
         'semTypes': [],
         'sentenceCount': {'annotated': ..., 'total': ...},
         'status': 'FN1_Sent'}

        :param fn_luid: The id number of the desired LU
        :type fn_luid: int
        :return: Basic information about the lexical unit
        :rtype: dict
        """
        return self.lu(fn_luid, ignorekeys=['subCorpus', 'exemplars'])

    def lu(self, fn_luid, ignorekeys=[], luName=None, frameID=None, frameName=None):
        """
        Access a lexical unit by its ID. luName, frameID, and frameName are used
        only in the event that the LU does not have a file in the database
        (which is the case for LUs with "Problem" status); in this case,
        a placeholder LU is created which just contains its name, ID, and frame.


        Usage examples:

        >>> from nltk.corpus import framenet as fn
        >>> fn.lu(256).name
        'foresee.v'
        >>> fn.lu(256).definition
        'COD: be aware of beforehand; predict.'
        >>> fn.lu(256).frame.name
        'Expectation'
        >>> list(map(PrettyDict, fn.lu(256).lexemes))
        [{'POS': 'V', 'breakBefore': 'false', 'headword': 'false', 'name': 'foresee', 'order': 1}]

        >>> fn.lu(227).exemplars[23] # doctest: +NORMALIZE_WHITESPACE
        exemplar sentence (352962):
        [sentNo] 0
        [aPos] 59699508
        <BLANKLINE>
        [LU] (227) guess.v in Coming_to_believe
        <BLANKLINE>
        [frame] (23) Coming_to_believe
        <BLANKLINE>
        [annotationSet] 2 annotation sets
        <BLANKLINE>
        [POS] 18 tags
        <BLANKLINE>
        [POS_tagset] BNC
        <BLANKLINE>
        [GF] 3 relations
        <BLANKLINE>
        [PT] 3 phrases
        <BLANKLINE>
        [Other] 1 entry
        <BLANKLINE>
        [text] + [Target] + [FE]
        <BLANKLINE>
        When he was inside the house , Culley noticed the characteristic
                                                      ------------------
                                                      Content
        <BLANKLINE>
        he would n't have guessed at .
        --                ******* --
        Co                        C1 [Evidence:INI]
         (Co=Cognizer, C1=Content)
        <BLANKLINE>
        <BLANKLINE>

        The dict that is returned from this function will contain most of the
        following information about the LU. Note that some LUs do not contain
        all of these pieces of information - particularly 'totalAnnotated' and
        'incorporatedFE' may be missing in some LUs:

        - 'name'       : the name of the LU (e.g. 'merger.n')
        - 'definition' : textual definition of the LU
        - 'ID'         : the internal ID number of the LU
        - '_type'      : 'lu'
        - 'status'     : e.g. 'Created'
        - 'frame'      : Frame that this LU belongs to
        - 'POS'        : the part of speech of this LU (e.g. 'N')
        - 'totalAnnotated' : total number of examples annotated with this LU
        - 'incorporatedFE' : FE that incorporates this LU (e.g. 'Ailment')
        - 'sentenceCount'  : a dict with the following two keys:
                 - 'annotated': number of sentences annotated with this LU
                 - 'total'    : total number of sentences with this LU

        - 'lexemes'  : a list of dicts describing the lemma of this LU.
           Each dict in the list contains these keys:

           - 'POS'     : part of speech e.g. 'N'
           - 'name'    : either single-lexeme e.g. 'merger' or
                         multi-lexeme e.g. 'a little'
           - 'order': the order of the lexeme in the lemma (starting from 1)
           - 'headword': a boolean ('true' or 'false')
           - 'breakBefore': Can this lexeme be separated from the previous lexeme?
                Consider: "take over.v" as in::

                         Germany took over the Netherlands in 2 days.
                         Germany took the Netherlands over in 2 days.

                In this case, 'breakBefore' would be "true" for the lexeme
                "over". Contrast this with "take after.v" as in::

                         Mary takes after her grandmother.
                        *Mary takes her grandmother after.

                In this case, 'breakBefore' would be "false" for the lexeme "after"

        - 'lemmaID'    : Can be used to connect lemmas in different LUs
        - 'semTypes'   : a list of semantic type objects for this LU
        - 'subCorpus'  : a list of subcorpora
           - Each item in the list is a dict containing the following keys:
              - 'name' :
              - 'sentence' : a list of sentences in the subcorpus
                 - each item in the list is a dict with the following keys:
                    - 'ID':
                    - 'sentNo':
                    - 'text': the text of the sentence
                    - 'aPos':
                    - 'annotationSet': a list of annotation sets
                       - each item in the list is a dict with the following keys:
                          - 'ID':
                          - 'status':
                          - 'layer': a list of layers
                             - each layer is a dict containing the following keys:
                                - 'name': layer name (e.g. 'BNC')
                                - 'rank':
                                - 'label': a list of labels for the layer
                                   - each label is a dict containing the following keys:
                                      - 'start': start pos of label in sentence 'text' (0-based)
                                      - 'end': end pos of label in sentence 'text' (0-based)
                                      - 'name': name of label (e.g. 'NN1')

        Under the hood, this implementation looks up the lexical unit information
        in the *frame* definition file. That file does not contain
        corpus annotations, so the LU files will be accessed on demand if those are
        needed. In principle, valence patterns could be loaded here too,
        though these are not currently supported.

        :param fn_luid: The id number of the lexical unit
        :type fn_luid: int
        :param ignorekeys: The keys to ignore. These keys will not be
            included in the output. (optional)
        :type ignorekeys: list(str)
        :return: All information about the lexical unit
        :rtype: dict
        """
        if not self._lu_idx:
            self._buildluindex()
        OOV = object()
        luinfo = self._lu_idx.get(fn_luid, OOV)
        if luinfo is OOV:
            self._warn('LU ID not found: {} ({}) in {} ({})'.format(luName, fn_luid, frameName, frameID))
            luinfo = AttrDict({'_type': 'lu', 'ID': fn_luid, 'name': luName, 'frameID': frameID, 'status': 'Problem'})
            f = self.frame_by_id(luinfo.frameID)
            assert f.name == frameName, (f.name, frameName)
            luinfo['frame'] = f
            self._lu_idx[fn_luid] = luinfo
        elif '_type' not in luinfo:
            f = self.frame_by_id(luinfo.frameID)
            luinfo = self._lu_idx[fn_luid]
        if ignorekeys:
            return AttrDict({k: v for k, v in luinfo.items() if k not in ignorekeys})
        return luinfo

    def _lu_file(self, lu, ignorekeys=[]):
        """
        Augment the LU information that was loaded from the frame file
        with additional information from the LU file.
        """
        fn_luid = lu.ID
        fname = f'lu{fn_luid}.xml'
        locpath = os.path.join(f'{self._root}', self._lu_dir, fname)
        if not self._lu_idx:
            self._buildluindex()
        try:
            with XMLCorpusView(locpath, 'lexUnit') as view:
                elt = view[0]
        except OSError as e:
            raise FramenetError(f'Unknown LU id: {fn_luid}') from e
        lu2 = self._handle_lexunit_elt(elt, ignorekeys)
        lu.URL = self._fnweb_url + '/' + self._lu_dir + '/' + fname
        lu.subCorpus = lu2.subCorpus
        lu.exemplars = SpecialList('luexemplars', [sent for subc in lu.subCorpus for sent in subc.sentence])
        for sent in lu.exemplars:
            sent['LU'] = lu
            sent['frame'] = lu.frame
            for aset in sent.annotationSet:
                aset['LU'] = lu
                aset['frame'] = lu.frame
        return lu

    def _loadsemtypes(self):
        """Create the semantic types index."""
        self._semtypes = AttrDict()
        with XMLCorpusView(self.abspath('semTypes.xml'), 'semTypes/semType', self._handle_semtype_elt) as view:
            for st in view:
                n = st['name']
                a = st['abbrev']
                i = st['ID']
                self._semtypes[n] = i
                self._semtypes[a] = i
                self._semtypes[i] = st
        roots = []
        for st in self.semtypes():
            if st.superType:
                st.superType = self.semtype(st.superType.supID)
                st.superType.subTypes.append(st)
            else:
                if st not in roots:
                    roots.append(st)
                st.rootType = st
        queue = list(roots)
        assert queue
        while queue:
            st = queue.pop(0)
            for child in st.subTypes:
                child.rootType = st.rootType
                queue.append(child)

    def propagate_semtypes(self):
        """
        Apply inference rules to distribute semtypes over relations between FEs.
        For FrameNet 1.5, this results in 1011 semtypes being propagated.
        (Not done by default because it requires loading all frame files,
        which takes several seconds. If this needed to be fast, it could be rewritten
        to traverse the neighboring relations on demand for each FE semtype.)

        >>> from nltk.corpus import framenet as fn
        >>> x = sum(1 for f in fn.frames() for fe in f.FE.values() if fe.semType)
        >>> fn.propagate_semtypes()
        >>> y = sum(1 for f in fn.frames() for fe in f.FE.values() if fe.semType)
        >>> y-x > 1000
        True
        """
        if not self._semtypes:
            self._loadsemtypes()
        if not self._ferel_idx:
            self._buildrelationindex()
        changed = True
        i = 0
        nPropagations = 0
        while changed:
            i += 1
            changed = False
            for ferel in self.fe_relations():
                superST = ferel.superFE.semType
                subST = ferel.subFE.semType
                try:
                    if superST and superST is not subST:
                        assert subST is None or self.semtype_inherits(subST, superST), (superST.name, ferel, subST.name)
                        if subST is None:
                            ferel.subFE.semType = subST = superST
                            changed = True
                            nPropagations += 1
                    if ferel.type.name in ['Perspective_on', 'Subframe', 'Precedes'] and subST and (subST is not superST):
                        assert superST is None, (superST.name, ferel, subST.name)
                        ferel.superFE.semType = superST = subST
                        changed = True
                        nPropagations += 1
                except AssertionError as ex:
                    continue

    def semtype(self, key):
        """
        >>> from nltk.corpus import framenet as fn
        >>> fn.semtype(233).name
        'Temperature'
        >>> fn.semtype(233).abbrev
        'Temp'
        >>> fn.semtype('Temperature').ID
        233

        :param key: The name, abbreviation, or id number of the semantic type
        :type key: string or int
        :return: Information about a semantic type
        :rtype: dict
        """
        if isinstance(key, int):
            stid = key
        else:
            try:
                stid = self._semtypes[key]
            except TypeError:
                self._loadsemtypes()
                stid = self._semtypes[key]
        try:
            st = self._semtypes[stid]
        except TypeError:
            self._loadsemtypes()
            st = self._semtypes[stid]
        return st

    def semtype_inherits(self, st, superST):
        if not isinstance(st, dict):
            st = self.semtype(st)
        if not isinstance(superST, dict):
            superST = self.semtype(superST)
        par = st.superType
        while par:
            if par is superST:
                return True
            par = par.superType
        return False

    def frames(self, name=None):
        """
        Obtain details for a specific frame.

        >>> from nltk.corpus import framenet as fn
        >>> len(fn.frames()) in (1019, 1221)    # FN 1.5 and 1.7, resp.
        True
        >>> x = PrettyList(fn.frames(r'(?i)crim'), maxReprSize=0, breakLines=True)
        >>> x.sort(key=itemgetter('ID'))
        >>> x
        [<frame ID=200 name=Criminal_process>,
         <frame ID=500 name=Criminal_investigation>,
         <frame ID=692 name=Crime_scenario>,
         <frame ID=700 name=Committing_crime>]

        A brief intro to Frames (excerpted from "FrameNet II: Extended
        Theory and Practice" by Ruppenhofer et. al., 2010):

        A Frame is a script-like conceptual structure that describes a
        particular type of situation, object, or event along with the
        participants and props that are needed for that Frame. For
        example, the "Apply_heat" frame describes a common situation
        involving a Cook, some Food, and a Heating_Instrument, and is
        evoked by words such as bake, blanch, boil, broil, brown,
        simmer, steam, etc.

        We call the roles of a Frame "frame elements" (FEs) and the
        frame-evoking words are called "lexical units" (LUs).

        FrameNet includes relations between Frames. Several types of
        relations are defined, of which the most important are:

           - Inheritance: An IS-A relation. The child frame is a subtype
             of the parent frame, and each FE in the parent is bound to
             a corresponding FE in the child. An example is the
             "Revenge" frame which inherits from the
             "Rewards_and_punishments" frame.

           - Using: The child frame presupposes the parent frame as
             background, e.g the "Speed" frame "uses" (or presupposes)
             the "Motion" frame; however, not all parent FEs need to be
             bound to child FEs.

           - Subframe: The child frame is a subevent of a complex event
             represented by the parent, e.g. the "Criminal_process" frame
             has subframes of "Arrest", "Arraignment", "Trial", and
             "Sentencing".

           - Perspective_on: The child frame provides a particular
             perspective on an un-perspectivized parent frame. A pair of
             examples consists of the "Hiring" and "Get_a_job" frames,
             which perspectivize the "Employment_start" frame from the
             Employer's and the Employee's point of view, respectively.

        :param name: A regular expression pattern used to match against
            Frame names. If 'name' is None, then a list of all
            Framenet Frames will be returned.
        :type name: str
        :return: A list of matching Frames (or all Frames).
        :rtype: list(AttrDict)
        """
        try:
            fIDs = list(self._frame_idx.keys())
        except AttributeError:
            self._buildframeindex()
            fIDs = list(self._frame_idx.keys())
        if name is not None:
            return PrettyList((self.frame(fID) for fID, finfo in self.frame_ids_and_names(name).items()))
        else:
            return PrettyLazyMap(self.frame, fIDs)

    def frame_ids_and_names(self, name=None):
        """
        Uses the frame index, which is much faster than looking up each frame definition
        if only the names and IDs are needed.
        """
        if not self._frame_idx:
            self._buildframeindex()
        return {fID: finfo.name for fID, finfo in self._frame_idx.items() if name is None or re.search(name, finfo.name) is not None}

    def fes(self, name=None, frame=None):
        """
        Lists frame element objects. If 'name' is provided, this is treated as
        a case-insensitive regular expression to filter by frame name.
        (Case-insensitivity is because casing of frame element names is not always
        consistent across frames.) Specify 'frame' to filter by a frame name pattern,
        ID, or object.

        >>> from nltk.corpus import framenet as fn
        >>> fn.fes('Noise_maker')
        [<fe ID=6043 name=Noise_maker>]
        >>> sorted([(fe.frame.name,fe.name) for fe in fn.fes('sound')]) # doctest: +NORMALIZE_WHITESPACE
        [('Cause_to_make_noise', 'Sound_maker'), ('Make_noise', 'Sound'),
         ('Make_noise', 'Sound_source'), ('Sound_movement', 'Location_of_sound_source'),
         ('Sound_movement', 'Sound'), ('Sound_movement', 'Sound_source'),
         ('Sounds', 'Component_sound'), ('Sounds', 'Location_of_sound_source'),
         ('Sounds', 'Sound_source'), ('Vocalizations', 'Location_of_sound_source'),
         ('Vocalizations', 'Sound_source')]
        >>> sorted([(fe.frame.name,fe.name) for fe in fn.fes('sound',r'(?i)make_noise')]) # doctest: +NORMALIZE_WHITESPACE
        [('Cause_to_make_noise', 'Sound_maker'),
         ('Make_noise', 'Sound'),
         ('Make_noise', 'Sound_source')]
        >>> sorted(set(fe.name for fe in fn.fes('^sound')))
        ['Sound', 'Sound_maker', 'Sound_source']
        >>> len(fn.fes('^sound$'))
        2

        :param name: A regular expression pattern used to match against
            frame element names. If 'name' is None, then a list of all
            frame elements will be returned.
        :type name: str
        :return: A list of matching frame elements
        :rtype: list(AttrDict)
        """
        if frame is not None:
            if isinstance(frame, int):
                frames = [self.frame(frame)]
            elif isinstance(frame, str):
                frames = self.frames(frame)
            else:
                frames = [frame]
        else:
            frames = self.frames()
        return PrettyList((fe for f in frames for fename, fe in f.FE.items() if name is None or re.search(name, fename, re.I)))

    def lus(self, name=None, frame=None):
        """
        Obtain details for lexical units.
        Optionally restrict by lexical unit name pattern, and/or to a certain frame
        or frames whose name matches a pattern.

        >>> from nltk.corpus import framenet as fn
        >>> len(fn.lus()) in (11829, 13572) # FN 1.5 and 1.7, resp.
        True
        >>> PrettyList(sorted(fn.lus(r'(?i)a little'), key=itemgetter('ID')), maxReprSize=0, breakLines=True)
        [<lu ID=14733 name=a little.n>,
         <lu ID=14743 name=a little.adv>,
         <lu ID=14744 name=a little bit.adv>]
        >>> PrettyList(sorted(fn.lus(r'interest', r'(?i)stimulus'), key=itemgetter('ID')))
        [<lu ID=14894 name=interested.a>, <lu ID=14920 name=interesting.a>]

        A brief intro to Lexical Units (excerpted from "FrameNet II:
        Extended Theory and Practice" by Ruppenhofer et. al., 2010):

        A lexical unit (LU) is a pairing of a word with a meaning. For
        example, the "Apply_heat" Frame describes a common situation
        involving a Cook, some Food, and a Heating Instrument, and is
        _evoked_ by words such as bake, blanch, boil, broil, brown,
        simmer, steam, etc. These frame-evoking words are the LUs in the
        Apply_heat frame. Each sense of a polysemous word is a different
        LU.

        We have used the word "word" in talking about LUs. The reality
        is actually rather complex. When we say that the word "bake" is
        polysemous, we mean that the lemma "bake.v" (which has the
        word-forms "bake", "bakes", "baked", and "baking") is linked to
        three different frames:

           - Apply_heat: "Michelle baked the potatoes for 45 minutes."

           - Cooking_creation: "Michelle baked her mother a cake for her birthday."

           - Absorb_heat: "The potatoes have to bake for more than 30 minutes."

        These constitute three different LUs, with different
        definitions.

        Multiword expressions such as "given name" and hyphenated words
        like "shut-eye" can also be LUs. Idiomatic phrases such as
        "middle of nowhere" and "give the slip (to)" are also defined as
        LUs in the appropriate frames ("Isolated_places" and "Evading",
        respectively), and their internal structure is not analyzed.

        Framenet provides multiple annotated examples of each sense of a
        word (i.e. each LU).  Moreover, the set of examples
        (approximately 20 per LU) illustrates all of the combinatorial
        possibilities of the lexical unit.

        Each LU is linked to a Frame, and hence to the other words which
        evoke that Frame. This makes the FrameNet database similar to a
        thesaurus, grouping together semantically similar words.

        In the simplest case, frame-evoking words are verbs such as
        "fried" in:

           "Matilde fried the catfish in a heavy iron skillet."

        Sometimes event nouns may evoke a Frame. For example,
        "reduction" evokes "Cause_change_of_scalar_position" in:

           "...the reduction of debt levels to $665 million from $2.6 billion."

        Adjectives may also evoke a Frame. For example, "asleep" may
        evoke the "Sleep" frame as in:

           "They were asleep for hours."

        Many common nouns, such as artifacts like "hat" or "tower",
        typically serve as dependents rather than clearly evoking their
        own frames.

        :param name: A regular expression pattern used to search the LU
            names. Note that LU names take the form of a dotted
            string (e.g. "run.v" or "a little.adv") in which a
            lemma precedes the "." and a POS follows the
            dot. The lemma may be composed of a single lexeme
            (e.g. "run") or of multiple lexemes (e.g. "a
            little"). If 'name' is not given, then all LUs will
            be returned.

            The valid POSes are:

                   v    - verb
                   n    - noun
                   a    - adjective
                   adv  - adverb
                   prep - preposition
                   num  - numbers
                   intj - interjection
                   art  - article
                   c    - conjunction
                   scon - subordinating conjunction

        :type name: str
        :type frame: str or int or frame
        :return: A list of selected (or all) lexical units
        :rtype: list of LU objects (dicts). See the lu() function for info
          about the specifics of LU objects.

        """
        if not self._lu_idx:
            self._buildluindex()
        if name is not None:
            result = PrettyList((self.lu(luID) for luID, luName in self.lu_ids_and_names(name).items()))
            if frame is not None:
                if isinstance(frame, int):
                    frameIDs = {frame}
                elif isinstance(frame, str):
                    frameIDs = {f.ID for f in self.frames(frame)}
                else:
                    frameIDs = {frame.ID}
                result = PrettyList((lu for lu in result if lu.frame.ID in frameIDs))
        elif frame is not None:
            if isinstance(frame, int):
                frames = [self.frame(frame)]
            elif isinstance(frame, str):
                frames = self.frames(frame)
            else:
                frames = [frame]
            result = PrettyLazyIteratorList(iter(LazyConcatenation((list(f.lexUnit.values()) for f in frames))))
        else:
            luIDs = [luID for luID, lu in self._lu_idx.items() if lu.status not in self._bad_statuses]
            result = PrettyLazyMap(self.lu, luIDs)
        return result

    def lu_ids_and_names(self, name=None):
        """
        Uses the LU index, which is much faster than looking up each LU definition
        if only the names and IDs are needed.
        """
        if not self._lu_idx:
            self._buildluindex()
        return {luID: luinfo.name for luID, luinfo in self._lu_idx.items() if luinfo.status not in self._bad_statuses and (name is None or re.search(name, luinfo.name) is not None)}

    def docs_metadata(self, name=None):
        """
        Return an index of the annotated documents in Framenet.

        Details for a specific annotated document can be obtained using this
        class's doc() function and pass it the value of the 'ID' field.

        >>> from nltk.corpus import framenet as fn
        >>> len(fn.docs()) in (78, 107) # FN 1.5 and 1.7, resp.
        True
        >>> set([x.corpname for x in fn.docs_metadata()])>=set(['ANC', 'KBEval',                     'LUCorpus-v0.3', 'Miscellaneous', 'NTI', 'PropBank'])
        True

        :param name: A regular expression pattern used to search the
            file name of each annotated document. The document's
            file name contains the name of the corpus that the
            document is from, followed by two underscores "__"
            followed by the document name. So, for example, the
            file name "LUCorpus-v0.3__20000410_nyt-NEW.xml" is
            from the corpus named "LUCorpus-v0.3" and the
            document name is "20000410_nyt-NEW.xml".
        :type name: str
        :return: A list of selected (or all) annotated documents
        :rtype: list of dicts, where each dict object contains the following
                keys:

                - 'name'
                - 'ID'
                - 'corpid'
                - 'corpname'
                - 'description'
                - 'filename'
        """
        try:
            ftlist = PrettyList(self._fulltext_idx.values())
        except AttributeError:
            self._buildcorpusindex()
            ftlist = PrettyList(self._fulltext_idx.values())
        if name is None:
            return ftlist
        else:
            return PrettyList((x for x in ftlist if re.search(name, x['filename']) is not None))

    def docs(self, name=None):
        """
        Return a list of the annotated full-text documents in FrameNet,
        optionally filtered by a regex to be matched against the document name.
        """
        return PrettyLazyMap(lambda x: self.doc(x.ID), self.docs_metadata(name))

    def sents(self, exemplars=True, full_text=True):
        """
        Annotated sentences matching the specified criteria.
        """
        if exemplars:
            if full_text:
                return self.exemplars() + self.ft_sents()
            else:
                return self.exemplars()
        elif full_text:
            return self.ft_sents()

    def annotations(self, luNamePattern=None, exemplars=True, full_text=True):
        """
        Frame annotation sets matching the specified criteria.
        """
        if exemplars:
            epart = PrettyLazyIteratorList((sent.frameAnnotation for sent in self.exemplars(luNamePattern)))
        else:
            epart = []
        if full_text:
            if luNamePattern is not None:
                matchedLUIDs = set(self.lu_ids_and_names(luNamePattern).keys())
            ftpart = PrettyLazyIteratorList((aset for sent in self.ft_sents() for aset in sent.annotationSet[1:] if luNamePattern is None or aset.get('luID', 'CXN_ASET') in matchedLUIDs))
        else:
            ftpart = []
        if exemplars:
            if full_text:
                return epart + ftpart
            else:
                return epart
        elif full_text:
            return ftpart

    def exemplars(self, luNamePattern=None, frame=None, fe=None, fe2=None):
        """
        Lexicographic exemplar sentences, optionally filtered by LU name and/or 1-2 FEs that
        are realized overtly. 'frame' may be a name pattern, frame ID, or frame instance.
        'fe' may be a name pattern or FE instance; if specified, 'fe2' may also
        be specified to retrieve sentences with both overt FEs (in either order).
        """
        if fe is None and fe2 is not None:
            raise FramenetError('exemplars(..., fe=None, fe2=<value>) is not allowed')
        elif fe is not None and fe2 is not None:
            if not isinstance(fe2, str):
                if isinstance(fe, str):
                    fe, fe2 = (fe2, fe)
                elif fe.frame is not fe2.frame:
                    raise FramenetError('exemplars() call with inconsistent `fe` and `fe2` specification (frames must match)')
        if frame is None and fe is not None and (not isinstance(fe, str)):
            frame = fe.frame
        lusByFrame = defaultdict(list)
        if frame is not None or luNamePattern is not None:
            if frame is None or isinstance(frame, str):
                if luNamePattern is not None:
                    frames = set()
                    for lu in self.lus(luNamePattern, frame=frame):
                        frames.add(lu.frame.ID)
                        lusByFrame[lu.frame.name].append(lu)
                    frames = LazyMap(self.frame, list(frames))
                else:
                    frames = self.frames(frame)
            else:
                if isinstance(frame, int):
                    frames = [self.frame(frame)]
                else:
                    frames = [frame]
                if luNamePattern is not None:
                    lusByFrame = {frame.name: self.lus(luNamePattern, frame=frame)}
            if fe is not None:
                if isinstance(fe, str):
                    frames = PrettyLazyIteratorList((f for f in frames if fe in f.FE or any((re.search(fe, ffe, re.I) for ffe in f.FE.keys()))))
                else:
                    if fe.frame not in frames:
                        raise FramenetError('exemplars() call with inconsistent `frame` and `fe` specification')
                    frames = [fe.frame]
                if fe2 is not None:
                    if isinstance(fe2, str):
                        frames = PrettyLazyIteratorList((f for f in frames if fe2 in f.FE or any((re.search(fe2, ffe, re.I) for ffe in f.FE.keys()))))
        elif fe is not None:
            frames = {ffe.frame.ID for ffe in self.fes(fe)}
            if fe2 is not None:
                frames2 = {ffe.frame.ID for ffe in self.fes(fe2)}
                frames = frames & frames2
            frames = LazyMap(self.frame, list(frames))
        else:
            frames = self.frames()

        def _matching_exs():
            for f in frames:
                fes = fes2 = None
                if fe is not None:
                    fes = {ffe for ffe in f.FE.keys() if re.search(fe, ffe, re.I)} if isinstance(fe, str) else {fe.name}
                    if fe2 is not None:
                        fes2 = {ffe for ffe in f.FE.keys() if re.search(fe2, ffe, re.I)} if isinstance(fe2, str) else {fe2.name}
                for lu in lusByFrame[f.name] if luNamePattern is not None else f.lexUnit.values():
                    for ex in lu.exemplars:
                        if (fes is None or self._exemplar_of_fes(ex, fes)) and (fes2 is None or self._exemplar_of_fes(ex, fes2)):
                            yield ex
        return PrettyLazyIteratorList(_matching_exs())

    def _exemplar_of_fes(self, ex, fes=None):
        """
        Given an exemplar sentence and a set of FE names, return the subset of FE names
        that are realized overtly in the sentence on the FE, FE2, or FE3 layer.

        If 'fes' is None, returns all overt FE names.
        """
        overtNames = set(list(zip(*ex.FE[0]))[2]) if ex.FE[0] else set()
        if 'FE2' in ex:
            overtNames |= set(list(zip(*ex.FE2[0]))[2]) if ex.FE2[0] else set()
            if 'FE3' in ex:
                overtNames |= set(list(zip(*ex.FE3[0]))[2]) if ex.FE3[0] else set()
        return overtNames & fes if fes is not None else overtNames

    def ft_sents(self, docNamePattern=None):
        """
        Full-text annotation sentences, optionally filtered by document name.
        """
        return PrettyLazyIteratorList((sent for d in self.docs(docNamePattern) for sent in d.sentence))

    def frame_relation_types(self):
        """
        Obtain a list of frame relation types.

        >>> from nltk.corpus import framenet as fn
        >>> frts = sorted(fn.frame_relation_types(), key=itemgetter('ID'))
        >>> isinstance(frts, list)
        True
        >>> len(frts) in (9, 10)    # FN 1.5 and 1.7, resp.
        True
        >>> PrettyDict(frts[0], breakLines=True)
        {'ID': 1,
         '_type': 'framerelationtype',
         'frameRelations': [<Parent=Event -- Inheritance -> Child=Change_of_consistency>, <Parent=Event -- Inheritance -> Child=Rotting>, ...],
         'name': 'Inheritance',
         'subFrameName': 'Child',
         'superFrameName': 'Parent'}

        :return: A list of all of the frame relation types in framenet
        :rtype: list(dict)
        """
        if not self._freltyp_idx:
            self._buildrelationindex()
        return self._freltyp_idx.values()

    def frame_relations(self, frame=None, frame2=None, type=None):
        """
        :param frame: (optional) frame object, name, or ID; only relations involving
            this frame will be returned
        :param frame2: (optional; 'frame' must be a different frame) only show relations
            between the two specified frames, in either direction
        :param type: (optional) frame relation type (name or object); show only relations
            of this type
        :type frame: int or str or AttrDict
        :return: A list of all of the frame relations in framenet
        :rtype: list(dict)

        >>> from nltk.corpus import framenet as fn
        >>> frels = fn.frame_relations()
        >>> isinstance(frels, list)
        True
        >>> len(frels) in (1676, 2070)  # FN 1.5 and 1.7, resp.
        True
        >>> PrettyList(fn.frame_relations('Cooking_creation'), maxReprSize=0, breakLines=True)
        [<Parent=Intentionally_create -- Inheritance -> Child=Cooking_creation>,
         <Parent=Apply_heat -- Using -> Child=Cooking_creation>,
         <MainEntry=Apply_heat -- See_also -> ReferringEntry=Cooking_creation>]
        >>> PrettyList(fn.frame_relations(274), breakLines=True)
        [<Parent=Avoiding -- Inheritance -> Child=Dodging>,
         <Parent=Avoiding -- Inheritance -> Child=Evading>, ...]
        >>> PrettyList(fn.frame_relations(fn.frame('Cooking_creation')), breakLines=True)
        [<Parent=Intentionally_create -- Inheritance -> Child=Cooking_creation>,
         <Parent=Apply_heat -- Using -> Child=Cooking_creation>, ...]
        >>> PrettyList(fn.frame_relations('Cooking_creation', type='Inheritance'))
        [<Parent=Intentionally_create -- Inheritance -> Child=Cooking_creation>]
        >>> PrettyList(fn.frame_relations('Cooking_creation', 'Apply_heat'), breakLines=True) # doctest: +NORMALIZE_WHITESPACE
        [<Parent=Apply_heat -- Using -> Child=Cooking_creation>,
        <MainEntry=Apply_heat -- See_also -> ReferringEntry=Cooking_creation>]
        """
        relation_type = type
        if not self._frel_idx:
            self._buildrelationindex()
        rels = None
        if relation_type is not None:
            if not isinstance(relation_type, dict):
                type = [rt for rt in self.frame_relation_types() if rt.name == type][0]
                assert isinstance(type, dict)
        if frame is not None:
            if isinstance(frame, dict) and 'frameRelations' in frame:
                rels = PrettyList(frame.frameRelations)
            else:
                if not isinstance(frame, int):
                    if isinstance(frame, dict):
                        frame = frame.ID
                    else:
                        frame = self.frame_by_name(frame).ID
                rels = [self._frel_idx[frelID] for frelID in self._frel_f_idx[frame]]
            if type is not None:
                rels = [rel for rel in rels if rel.type is type]
        elif type is not None:
            rels = type.frameRelations
        else:
            rels = self._frel_idx.values()
        if frame2 is not None:
            if frame is None:
                raise FramenetError('frame_relations(frame=None, frame2=<value>) is not allowed')
            if not isinstance(frame2, int):
                if isinstance(frame2, dict):
                    frame2 = frame2.ID
                else:
                    frame2 = self.frame_by_name(frame2).ID
            if frame == frame2:
                raise FramenetError('The two frame arguments to frame_relations() must be different frames')
            rels = [rel for rel in rels if rel.superFrame.ID == frame2 or rel.subFrame.ID == frame2]
        return PrettyList(sorted(rels, key=lambda frel: (frel.type.ID, frel.superFrameName, frel.subFrameName)))

    def fe_relations(self):
        """
        Obtain a list of frame element relations.

        >>> from nltk.corpus import framenet as fn
        >>> ferels = fn.fe_relations()
        >>> isinstance(ferels, list)
        True
        >>> len(ferels) in (10020, 12393)   # FN 1.5 and 1.7, resp.
        True
        >>> PrettyDict(ferels[0], breakLines=True) # doctest: +NORMALIZE_WHITESPACE
        {'ID': 14642,
        '_type': 'ferelation',
        'frameRelation': <Parent=Abounding_with -- Inheritance -> Child=Lively_place>,
        'subFE': <fe ID=11370 name=Degree>,
        'subFEName': 'Degree',
        'subFrame': <frame ID=1904 name=Lively_place>,
        'subID': 11370,
        'supID': 2271,
        'superFE': <fe ID=2271 name=Degree>,
        'superFEName': 'Degree',
        'superFrame': <frame ID=262 name=Abounding_with>,
        'type': <framerelationtype ID=1 name=Inheritance>}

        :return: A list of all of the frame element relations in framenet
        :rtype: list(dict)
        """
        if not self._ferel_idx:
            self._buildrelationindex()
        return PrettyList(sorted(self._ferel_idx.values(), key=lambda ferel: (ferel.type.ID, ferel.frameRelation.superFrameName, ferel.superFEName, ferel.frameRelation.subFrameName, ferel.subFEName)))

    def semtypes(self):
        """
        Obtain a list of semantic types.

        >>> from nltk.corpus import framenet as fn
        >>> stypes = fn.semtypes()
        >>> len(stypes) in (73, 109) # FN 1.5 and 1.7, resp.
        True
        >>> sorted(stypes[0].keys())
        ['ID', '_type', 'abbrev', 'definition', 'definitionMarkup', 'name', 'rootType', 'subTypes', 'superType']

        :return: A list of all of the semantic types in framenet
        :rtype: list(dict)
        """
        if not self._semtypes:
            self._loadsemtypes()
        return PrettyList((self._semtypes[i] for i in self._semtypes if isinstance(i, int)))

    def _load_xml_attributes(self, d, elt):
        """
        Extracts a subset of the attributes from the given element and
        returns them in a dictionary.

        :param d: A dictionary in which to store the attributes.
        :type d: dict
        :param elt: An ElementTree Element
        :type elt: Element
        :return: Returns the input dict ``d`` possibly including attributes from ``elt``
        :rtype: dict
        """
        d = type(d)(d)
        try:
            attr_dict = elt.attrib
        except AttributeError:
            return d
        if attr_dict is None:
            return d
        ignore_attrs = ['xsi', 'schemaLocation', 'xmlns', 'bgColor', 'fgColor']
        for attr in attr_dict:
            if any((attr.endswith(x) for x in ignore_attrs)):
                continue
            val = attr_dict[attr]
            if val.isdigit():
                d[attr] = int(val)
            else:
                d[attr] = val
        return d

    def _strip_tags(self, data):
        """
        Gets rid of all tags and newline characters from the given input

        :return: A cleaned-up version of the input string
        :rtype: str
        """
        try:
            "\n            # Look for boundary issues in markup. (Sometimes FEs are pluralized in definitions.)\n            m = re.search(r'\\w[<][^/]|[<][/][^>]+[>](s\\w|[a-rt-z0-9])', data)\n            if m:\n                print('Markup boundary:', data[max(0,m.start(0)-10):m.end(0)+10].replace('\\n',' '), file=sys.stderr)\n            "
            data = data.replace('<t>', '')
            data = data.replace('</t>', '')
            data = re.sub('<fex name="[^"]+">', '', data)
            data = data.replace('</fex>', '')
            data = data.replace('<fen>', '')
            data = data.replace('</fen>', '')
            data = data.replace('<m>', '')
            data = data.replace('</m>', '')
            data = data.replace('<ment>', '')
            data = data.replace('</ment>', '')
            data = data.replace('<ex>', "'")
            data = data.replace('</ex>', "'")
            data = data.replace('<gov>', '')
            data = data.replace('</gov>', '')
            data = data.replace('<x>', '')
            data = data.replace('</x>', '')
            data = data.replace('<def-root>', '')
            data = data.replace('</def-root>', '')
            data = data.replace('\n', ' ')
        except AttributeError:
            pass
        return data

    def _handle_elt(self, elt, tagspec=None):
        """Extracts and returns the attributes of the given element"""
        return self._load_xml_attributes(AttrDict(), elt)

    def _handle_fulltextindex_elt(self, elt, tagspec=None):
        """
        Extracts corpus/document info from the fulltextIndex.xml file.

        Note that this function "flattens" the information contained
        in each of the "corpus" elements, so that each "document"
        element will contain attributes for the corpus and
        corpusid. Also, each of the "document" items will contain a
        new attribute called "filename" that is the base file name of
        the xml file for the document in the "fulltext" subdir of the
        Framenet corpus.
        """
        ftinfo = self._load_xml_attributes(AttrDict(), elt)
        corpname = ftinfo.name
        corpid = ftinfo.ID
        retlist = []
        for sub in elt:
            if sub.tag.endswith('document'):
                doc = self._load_xml_attributes(AttrDict(), sub)
                if 'name' in doc:
                    docname = doc.name
                else:
                    docname = doc.description
                doc.filename = f'{corpname}__{docname}.xml'
                doc.URL = self._fnweb_url + '/' + self._fulltext_dir + '/' + doc.filename
                doc.corpname = corpname
                doc.corpid = corpid
                retlist.append(doc)
        return retlist

    def _handle_frame_elt(self, elt, ignorekeys=[]):
        """Load the info for a Frame from a frame xml file"""
        frinfo = self._load_xml_attributes(AttrDict(), elt)
        frinfo['_type'] = 'frame'
        frinfo['definition'] = ''
        frinfo['definitionMarkup'] = ''
        frinfo['FE'] = PrettyDict()
        frinfo['FEcoreSets'] = []
        frinfo['lexUnit'] = PrettyDict()
        frinfo['semTypes'] = []
        for k in ignorekeys:
            if k in frinfo:
                del frinfo[k]
        for sub in elt:
            if sub.tag.endswith('definition') and 'definition' not in ignorekeys:
                frinfo['definitionMarkup'] = sub.text
                frinfo['definition'] = self._strip_tags(sub.text)
            elif sub.tag.endswith('FE') and 'FE' not in ignorekeys:
                feinfo = self._handle_fe_elt(sub)
                frinfo['FE'][feinfo.name] = feinfo
                feinfo['frame'] = frinfo
            elif sub.tag.endswith('FEcoreSet') and 'FEcoreSet' not in ignorekeys:
                coreset = self._handle_fecoreset_elt(sub)
                frinfo['FEcoreSets'].append(PrettyList((frinfo['FE'][fe.name] for fe in coreset)))
            elif sub.tag.endswith('lexUnit') and 'lexUnit' not in ignorekeys:
                luentry = self._handle_framelexunit_elt(sub)
                if luentry['status'] in self._bad_statuses:
                    continue
                luentry['frame'] = frinfo
                luentry['URL'] = self._fnweb_url + '/' + self._lu_dir + '/' + 'lu{}.xml'.format(luentry['ID'])
                luentry['subCorpus'] = Future((lambda lu: lambda: self._lu_file(lu).subCorpus)(luentry))
                luentry['exemplars'] = Future((lambda lu: lambda: self._lu_file(lu).exemplars)(luentry))
                frinfo['lexUnit'][luentry.name] = luentry
                if not self._lu_idx:
                    self._buildluindex()
                self._lu_idx[luentry.ID] = luentry
            elif sub.tag.endswith('semType') and 'semTypes' not in ignorekeys:
                semtypeinfo = self._load_xml_attributes(AttrDict(), sub)
                frinfo['semTypes'].append(self.semtype(semtypeinfo.ID))
        frinfo['frameRelations'] = self.frame_relations(frame=frinfo)
        for fe in frinfo.FE.values():
            if fe.requiresFE:
                name, ID = (fe.requiresFE.name, fe.requiresFE.ID)
                fe.requiresFE = frinfo.FE[name]
                assert fe.requiresFE.ID == ID
            if fe.excludesFE:
                name, ID = (fe.excludesFE.name, fe.excludesFE.ID)
                fe.excludesFE = frinfo.FE[name]
                assert fe.excludesFE.ID == ID
        return frinfo

    def _handle_fecoreset_elt(self, elt):
        """Load fe coreset info from xml."""
        info = self._load_xml_attributes(AttrDict(), elt)
        tmp = []
        for sub in elt:
            tmp.append(self._load_xml_attributes(AttrDict(), sub))
        return tmp

    def _handle_framerelationtype_elt(self, elt, *args):
        """Load frame-relation element and its child fe-relation elements from frRelation.xml."""
        info = self._load_xml_attributes(AttrDict(), elt)
        info['_type'] = 'framerelationtype'
        info['frameRelations'] = PrettyList()
        for sub in elt:
            if sub.tag.endswith('frameRelation'):
                frel = self._handle_framerelation_elt(sub)
                frel['type'] = info
                for ferel in frel.feRelations:
                    ferel['type'] = info
                info['frameRelations'].append(frel)
        return info

    def _handle_framerelation_elt(self, elt):
        """Load frame-relation element and its child fe-relation elements from frRelation.xml."""
        info = self._load_xml_attributes(AttrDict(), elt)
        assert info['superFrameName'] != info['subFrameName'], (elt, info)
        info['_type'] = 'framerelation'
        info['feRelations'] = PrettyList()
        for sub in elt:
            if sub.tag.endswith('FERelation'):
                ferel = self._handle_elt(sub)
                ferel['_type'] = 'ferelation'
                ferel['frameRelation'] = info
                info['feRelations'].append(ferel)
        return info

    def _handle_fulltextannotation_elt(self, elt):
        """Load full annotation info for a document from its xml
        file. The main element (fullTextAnnotation) contains a 'header'
        element (which we ignore here) and a bunch of 'sentence'
        elements."""
        info = AttrDict()
        info['_type'] = 'fulltext_annotation'
        info['sentence'] = []
        for sub in elt:
            if sub.tag.endswith('header'):
                continue
            elif sub.tag.endswith('sentence'):
                s = self._handle_fulltext_sentence_elt(sub)
                s.doc = info
                info['sentence'].append(s)
        return info

    def _handle_fulltext_sentence_elt(self, elt):
        """Load information from the given 'sentence' element. Each
        'sentence' element contains a "text" and "annotationSet" sub
        elements."""
        info = self._load_xml_attributes(AttrDict(), elt)
        info['_type'] = 'fulltext_sentence'
        info['annotationSet'] = []
        info['targets'] = []
        target_spans = set()
        info['_ascii'] = types.MethodType(_annotation_ascii, info)
        info['text'] = ''
        for sub in elt:
            if sub.tag.endswith('text'):
                info['text'] = self._strip_tags(sub.text)
            elif sub.tag.endswith('annotationSet'):
                a = self._handle_fulltextannotationset_elt(sub, is_pos=len(info['annotationSet']) == 0)
                if 'cxnID' in a:
                    continue
                a.sent = info
                a.text = info.text
                info['annotationSet'].append(a)
                if 'Target' in a:
                    for tspan in a.Target:
                        if tspan in target_spans:
                            self._warn('Duplicate target span "{}"'.format(info.text[slice(*tspan)]), tspan, 'in sentence', info['ID'], info.text)
                        else:
                            target_spans.add(tspan)
                    info['targets'].append((a.Target, a.luName, a.frameName))
        assert info['annotationSet'][0].status == 'UNANN'
        info['POS'] = info['annotationSet'][0].POS
        info['POS_tagset'] = info['annotationSet'][0].POS_tagset
        return info

    def _handle_fulltextannotationset_elt(self, elt, is_pos=False):
        """Load information from the given 'annotationSet' element. Each
        'annotationSet' contains several "layer" elements."""
        info = self._handle_luannotationset_elt(elt, is_pos=is_pos)
        if not is_pos:
            info['_type'] = 'fulltext_annotationset'
            if 'cxnID' not in info:
                info['LU'] = self.lu(info.luID, luName=info.luName, frameID=info.frameID, frameName=info.frameName)
                info['frame'] = info.LU.frame
        return info

    def _handle_fulltextlayer_elt(self, elt):
        """Load information from the given 'layer' element. Each
        'layer' contains several "label" elements."""
        info = self._load_xml_attributes(AttrDict(), elt)
        info['_type'] = 'layer'
        info['label'] = []
        for sub in elt:
            if sub.tag.endswith('label'):
                l = self._load_xml_attributes(AttrDict(), sub)
                info['label'].append(l)
        return info

    def _handle_framelexunit_elt(self, elt):
        """Load the lexical unit info from an xml element in a frame's xml file."""
        luinfo = AttrDict()
        luinfo['_type'] = 'lu'
        luinfo = self._load_xml_attributes(luinfo, elt)
        luinfo['definition'] = ''
        luinfo['definitionMarkup'] = ''
        luinfo['sentenceCount'] = PrettyDict()
        luinfo['lexemes'] = PrettyList()
        luinfo['semTypes'] = PrettyList()
        for sub in elt:
            if sub.tag.endswith('definition'):
                luinfo['definitionMarkup'] = sub.text
                luinfo['definition'] = self._strip_tags(sub.text)
            elif sub.tag.endswith('sentenceCount'):
                luinfo['sentenceCount'] = self._load_xml_attributes(PrettyDict(), sub)
            elif sub.tag.endswith('lexeme'):
                lexemeinfo = self._load_xml_attributes(PrettyDict(), sub)
                if not isinstance(lexemeinfo.name, str):
                    lexemeinfo.name = str(lexemeinfo.name)
                luinfo['lexemes'].append(lexemeinfo)
            elif sub.tag.endswith('semType'):
                semtypeinfo = self._load_xml_attributes(PrettyDict(), sub)
                luinfo['semTypes'].append(self.semtype(semtypeinfo.ID))
        luinfo['lexemes'].sort(key=lambda x: x.order)
        return luinfo

    def _handle_lexunit_elt(self, elt, ignorekeys):
        """
        Load full info for a lexical unit from its xml file.
        This should only be called when accessing corpus annotations
        (which are not included in frame files).
        """
        luinfo = self._load_xml_attributes(AttrDict(), elt)
        luinfo['_type'] = 'lu'
        luinfo['definition'] = ''
        luinfo['definitionMarkup'] = ''
        luinfo['subCorpus'] = PrettyList()
        luinfo['lexemes'] = PrettyList()
        luinfo['semTypes'] = PrettyList()
        for k in ignorekeys:
            if k in luinfo:
                del luinfo[k]
        for sub in elt:
            if sub.tag.endswith('header'):
                continue
            elif sub.tag.endswith('valences'):
                continue
            elif sub.tag.endswith('definition') and 'definition' not in ignorekeys:
                luinfo['definitionMarkup'] = sub.text
                luinfo['definition'] = self._strip_tags(sub.text)
            elif sub.tag.endswith('subCorpus') and 'subCorpus' not in ignorekeys:
                sc = self._handle_lusubcorpus_elt(sub)
                if sc is not None:
                    luinfo['subCorpus'].append(sc)
            elif sub.tag.endswith('lexeme') and 'lexeme' not in ignorekeys:
                luinfo['lexemes'].append(self._load_xml_attributes(PrettyDict(), sub))
            elif sub.tag.endswith('semType') and 'semType' not in ignorekeys:
                semtypeinfo = self._load_xml_attributes(AttrDict(), sub)
                luinfo['semTypes'].append(self.semtype(semtypeinfo.ID))
        return luinfo

    def _handle_lusubcorpus_elt(self, elt):
        """Load a subcorpus of a lexical unit from the given xml."""
        sc = AttrDict()
        try:
            sc['name'] = elt.get('name')
        except AttributeError:
            return None
        sc['_type'] = 'lusubcorpus'
        sc['sentence'] = []
        for sub in elt:
            if sub.tag.endswith('sentence'):
                s = self._handle_lusentence_elt(sub)
                if s is not None:
                    sc['sentence'].append(s)
        return sc

    def _handle_lusentence_elt(self, elt):
        """Load a sentence from a subcorpus of an LU from xml."""
        info = self._load_xml_attributes(AttrDict(), elt)
        info['_type'] = 'lusentence'
        info['annotationSet'] = []
        info['_ascii'] = types.MethodType(_annotation_ascii, info)
        for sub in elt:
            if sub.tag.endswith('text'):
                info['text'] = self._strip_tags(sub.text)
            elif sub.tag.endswith('annotationSet'):
                annset = self._handle_luannotationset_elt(sub, is_pos=len(info['annotationSet']) == 0)
                if annset is not None:
                    assert annset.status == 'UNANN' or 'FE' in annset, annset
                    if annset.status != 'UNANN':
                        info['frameAnnotation'] = annset
                    for k in ('Target', 'FE', 'FE2', 'FE3', 'GF', 'PT', 'POS', 'POS_tagset', 'Other', 'Sent', 'Verb', 'Noun', 'Adj', 'Adv', 'Prep', 'Scon', 'Art'):
                        if k in annset:
                            info[k] = annset[k]
                    info['annotationSet'].append(annset)
                    annset['sent'] = info
                    annset['text'] = info.text
        return info

    def _handle_luannotationset_elt(self, elt, is_pos=False):
        """Load an annotation set from a sentence in an subcorpus of an LU"""
        info = self._load_xml_attributes(AttrDict(), elt)
        info['_type'] = 'posannotationset' if is_pos else 'luannotationset'
        info['layer'] = []
        info['_ascii'] = types.MethodType(_annotation_ascii, info)
        if 'cxnID' in info:
            return info
        for sub in elt:
            if sub.tag.endswith('layer'):
                l = self._handle_lulayer_elt(sub)
                if l is not None:
                    overt = []
                    ni = {}
                    info['layer'].append(l)
                    for lbl in l.label:
                        if 'start' in lbl:
                            thespan = (lbl.start, lbl.end + 1, lbl.name)
                            if l.name not in ('Sent', 'Other'):
                                assert thespan not in overt, (info.ID, l.name, thespan)
                            overt.append(thespan)
                        elif lbl.name in ni:
                            self._warn('FE with multiple NI entries:', lbl.name, ni[lbl.name], lbl.itype)
                        else:
                            ni[lbl.name] = lbl.itype
                    overt = sorted(overt)
                    if l.name == 'Target':
                        if not overt:
                            self._warn('Skipping empty Target layer in annotation set ID={}'.format(info.ID))
                            continue
                        assert all((lblname == 'Target' for i, j, lblname in overt))
                        if 'Target' in info:
                            self._warn('Annotation set {} has multiple Target layers'.format(info.ID))
                        else:
                            info['Target'] = [(i, j) for i, j, _ in overt]
                    elif l.name == 'FE':
                        if l.rank == 1:
                            assert 'FE' not in info
                            info['FE'] = (overt, ni)
                        else:
                            assert 2 <= l.rank <= 3, l.rank
                            k = 'FE' + str(l.rank)
                            assert k not in info
                            info[k] = (overt, ni)
                    elif l.name in ('GF', 'PT'):
                        assert l.rank == 1
                        info[l.name] = overt
                    elif l.name in ('BNC', 'PENN'):
                        assert l.rank == 1
                        info['POS'] = overt
                        info['POS_tagset'] = l.name
                    else:
                        if is_pos:
                            if l.name not in ('NER', 'WSL'):
                                self._warn('Unexpected layer in sentence annotationset:', l.name)
                        elif l.name not in ('Sent', 'Verb', 'Noun', 'Adj', 'Adv', 'Prep', 'Scon', 'Art', 'Other'):
                            self._warn('Unexpected layer in frame annotationset:', l.name)
                        info[l.name] = overt
        if not is_pos and 'cxnID' not in info:
            if 'Target' not in info:
                self._warn(f'Missing target in annotation set ID={info.ID}')
            assert 'FE' in info
            if 'FE3' in info:
                assert 'FE2' in info
        return info

    def _handle_lulayer_elt(self, elt):
        """Load a layer from an annotation set"""
        layer = self._load_xml_attributes(AttrDict(), elt)
        layer['_type'] = 'lulayer'
        layer['label'] = []
        for sub in elt:
            if sub.tag.endswith('label'):
                l = self._load_xml_attributes(AttrDict(), sub)
                if l is not None:
                    layer['label'].append(l)
        return layer

    def _handle_fe_elt(self, elt):
        feinfo = self._load_xml_attributes(AttrDict(), elt)
        feinfo['_type'] = 'fe'
        feinfo['definition'] = ''
        feinfo['definitionMarkup'] = ''
        feinfo['semType'] = None
        feinfo['requiresFE'] = None
        feinfo['excludesFE'] = None
        for sub in elt:
            if sub.tag.endswith('definition'):
                feinfo['definitionMarkup'] = sub.text
                feinfo['definition'] = self._strip_tags(sub.text)
            elif sub.tag.endswith('semType'):
                stinfo = self._load_xml_attributes(AttrDict(), sub)
                feinfo['semType'] = self.semtype(stinfo.ID)
            elif sub.tag.endswith('requiresFE'):
                feinfo['requiresFE'] = self._load_xml_attributes(AttrDict(), sub)
            elif sub.tag.endswith('excludesFE'):
                feinfo['excludesFE'] = self._load_xml_attributes(AttrDict(), sub)
        return feinfo

    def _handle_semtype_elt(self, elt, tagspec=None):
        semt = self._load_xml_attributes(AttrDict(), elt)
        semt['_type'] = 'semtype'
        semt['superType'] = None
        semt['subTypes'] = PrettyList()
        for sub in elt:
            if sub.text is not None:
                semt['definitionMarkup'] = sub.text
                semt['definition'] = self._strip_tags(sub.text)
            else:
                supertypeinfo = self._load_xml_attributes(AttrDict(), sub)
                semt['superType'] = supertypeinfo
        return semt