from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, cast
from docutils import nodes
from docutils.nodes import Element, Node
from sphinx import addnodes
from sphinx.locale import __
from sphinx.util import logging, url_re
from sphinx.util.matching import Matcher
from sphinx.util.nodes import clean_astext, process_only_nodes
class TocTree:

    def __init__(self, env: 'BuildEnvironment') -> None:
        self.env = env

    def note(self, docname: str, toctreenode: addnodes.toctree) -> None:
        """Note a TOC tree directive in a document and gather information about
        file relations from it.
        """
        if toctreenode['glob']:
            self.env.glob_toctrees.add(docname)
        if toctreenode.get('numbered'):
            self.env.numbered_toctrees.add(docname)
        includefiles = toctreenode['includefiles']
        for includefile in includefiles:
            self.env.files_to_rebuild.setdefault(includefile, set()).add(docname)
        self.env.toctree_includes.setdefault(docname, []).extend(includefiles)

    def resolve(self, docname: str, builder: 'Builder', toctree: addnodes.toctree, prune: bool=True, maxdepth: int=0, titles_only: bool=False, collapse: bool=False, includehidden: bool=False) -> Optional[Element]:
        """Resolve a *toctree* node into individual bullet lists with titles
        as items, returning None (if no containing titles are found) or
        a new node.

        If *prune* is True, the tree is pruned to *maxdepth*, or if that is 0,
        to the value of the *maxdepth* option on the *toctree* node.
        If *titles_only* is True, only toplevel document titles will be in the
        resulting tree.
        If *collapse* is True, all branches not containing docname will
        be collapsed.
        """
        if toctree.get('hidden', False) and (not includehidden):
            return None
        generated_docnames: Dict[str, Tuple[str, str, str]] = self.env.domains['std'].initial_data['labels'].copy()
        toctree_ancestors = self.get_toctree_ancestors(docname)
        included = Matcher(self.env.config.include_patterns)
        excluded = Matcher(self.env.config.exclude_patterns)

        def _toctree_add_classes(node: Element, depth: int) -> None:
            """Add 'toctree-l%d' and 'current' classes to the toctree."""
            for subnode in node.children:
                if isinstance(subnode, (addnodes.compact_paragraph, nodes.list_item)):
                    subnode['classes'].append('toctree-l%d' % (depth - 1))
                    _toctree_add_classes(subnode, depth)
                elif isinstance(subnode, nodes.bullet_list):
                    _toctree_add_classes(subnode, depth + 1)
                elif isinstance(subnode, nodes.reference):
                    if subnode['refuri'] == docname:
                        if not subnode['anchorname']:
                            branchnode: Element = subnode
                            while branchnode:
                                branchnode['classes'].append('current')
                                branchnode = branchnode.parent
                        if subnode.parent.parent.get('iscurrent'):
                            return
                        while subnode:
                            subnode['iscurrent'] = True
                            subnode = subnode.parent

        def _entries_from_toctree(toctreenode: addnodes.toctree, parents: List[str], separate: bool=False, subtree: bool=False) -> List[Element]:
            """Return TOC entries for a toctree node."""
            refs = [(e[0], e[1]) for e in toctreenode['entries']]
            entries: List[Element] = []
            for title, ref in refs:
                try:
                    refdoc = None
                    if url_re.match(ref):
                        if title is None:
                            title = ref
                        reference = nodes.reference('', '', *[nodes.Text(title)], internal=False, refuri=ref, anchorname='')
                        para = addnodes.compact_paragraph('', '', reference)
                        item = nodes.list_item('', para)
                        toc = nodes.bullet_list('', item)
                    elif ref == 'self':
                        ref = toctreenode['parent']
                        if not title:
                            title = clean_astext(self.env.titles[ref])
                        reference = nodes.reference('', '', *[nodes.Text(title)], internal=True, refuri=ref, anchorname='')
                        para = addnodes.compact_paragraph('', '', reference)
                        item = nodes.list_item('', para)
                        toc = nodes.bullet_list('', item)
                    elif ref in generated_docnames:
                        docname, _, sectionname = generated_docnames[ref]
                        if not title:
                            title = sectionname
                        reference = nodes.reference('', title, internal=True, refuri=docname, anchorname='')
                        para = addnodes.compact_paragraph('', '', reference)
                        item = nodes.list_item('', para)
                        toc = nodes.bullet_list('', item)
                    else:
                        if ref in parents:
                            logger.warning(__('circular toctree references detected, ignoring: %s <- %s'), ref, ' <- '.join(parents), location=ref, type='toc', subtype='circular')
                            continue
                        refdoc = ref
                        toc = self.env.tocs[ref].deepcopy()
                        maxdepth = self.env.metadata[ref].get('tocdepth', 0)
                        if ref not in toctree_ancestors or (prune and maxdepth > 0):
                            self._toctree_prune(toc, 2, maxdepth, collapse)
                        process_only_nodes(toc, builder.tags)
                        if title and toc.children and (len(toc.children) == 1):
                            child = toc.children[0]
                            for refnode in child.findall(nodes.reference):
                                if refnode['refuri'] == ref and (not refnode['anchorname']):
                                    refnode.children = [nodes.Text(title)]
                    if not toc.children:
                        logger.warning(__("toctree contains reference to document %r that doesn't have a title: no link will be generated"), ref, location=toctreenode)
                except KeyError:
                    if excluded(self.env.doc2path(ref, False)):
                        message = __('toctree contains reference to excluded document %r')
                    elif not included(self.env.doc2path(ref, False)):
                        message = __('toctree contains reference to non-included document %r')
                    else:
                        message = __('toctree contains reference to nonexisting document %r')
                    logger.warning(message, ref, location=toctreenode)
                else:
                    if titles_only:
                        children = cast(Iterable[nodes.Element], toc)
                        for toplevel in children:
                            if len(toplevel) > 1:
                                subtrees = list(toplevel.findall(addnodes.toctree))
                                if subtrees:
                                    toplevel[1][:] = subtrees
                                else:
                                    toplevel.pop(1)
                    for subtocnode in list(toc.findall(addnodes.toctree)):
                        if not (subtocnode.get('hidden', False) and (not includehidden)):
                            i = subtocnode.parent.index(subtocnode) + 1
                            for entry in _entries_from_toctree(subtocnode, [refdoc] + parents, subtree=True):
                                subtocnode.parent.insert(i, entry)
                                i += 1
                            subtocnode.parent.remove(subtocnode)
                    if separate:
                        entries.append(toc)
                    else:
                        children = cast(Iterable[nodes.Element], toc)
                        entries.extend(children)
            if not subtree and (not separate):
                ret = nodes.bullet_list()
                ret += entries
                return [ret]
            return entries
        maxdepth = maxdepth or toctree.get('maxdepth', -1)
        if not titles_only and toctree.get('titlesonly', False):
            titles_only = True
        if not includehidden and toctree.get('includehidden', False):
            includehidden = True
        tocentries = _entries_from_toctree(toctree, [], separate=False)
        if not tocentries:
            return None
        newnode = addnodes.compact_paragraph('', '')
        caption = toctree.attributes.get('caption')
        if caption:
            caption_node = nodes.title(caption, '', *[nodes.Text(caption)])
            caption_node.line = toctree.line
            caption_node.source = toctree.source
            caption_node.rawsource = toctree['rawcaption']
            if hasattr(toctree, 'uid'):
                caption_node.uid = toctree.uid
                del toctree.uid
            newnode += caption_node
        newnode.extend(tocentries)
        newnode['toctree'] = True
        _toctree_add_classes(newnode, 1)
        self._toctree_prune(newnode, 1, maxdepth if prune else 0, collapse)
        if isinstance(newnode[-1], nodes.Element) and len(newnode[-1]) == 0:
            return None
        for refnode in newnode.findall(nodes.reference):
            if not url_re.match(refnode['refuri']):
                refnode['refuri'] = builder.get_relative_uri(docname, refnode['refuri']) + refnode['anchorname']
        return newnode

    def get_toctree_ancestors(self, docname: str) -> List[str]:
        parent = {}
        for p, children in self.env.toctree_includes.items():
            for child in children:
                parent[child] = p
        ancestors: List[str] = []
        d = docname
        while d in parent and d not in ancestors:
            ancestors.append(d)
            d = parent[d]
        return ancestors

    def _toctree_prune(self, node: Element, depth: int, maxdepth: int, collapse: bool=False) -> None:
        """Utility: Cut a TOC at a specified depth."""
        for subnode in node.children[:]:
            if isinstance(subnode, (addnodes.compact_paragraph, nodes.list_item)):
                self._toctree_prune(subnode, depth, maxdepth, collapse)
            elif isinstance(subnode, nodes.bullet_list):
                if maxdepth > 0 and depth > maxdepth:
                    subnode.parent.replace(subnode, [])
                elif collapse and depth > 1 and ('iscurrent' not in subnode.parent):
                    subnode.parent.remove(subnode)
                else:
                    self._toctree_prune(subnode, depth + 1, maxdepth, collapse)

    def get_toc_for(self, docname: str, builder: 'Builder') -> Node:
        """Return a TOC nodetree -- for use on the same page only!"""
        tocdepth = self.env.metadata[docname].get('tocdepth', 0)
        try:
            toc = self.env.tocs[docname].deepcopy()
            self._toctree_prune(toc, 2, tocdepth)
        except KeyError:
            return nodes.paragraph()
        process_only_nodes(toc, builder.tags)
        for node in toc.findall(nodes.reference):
            node['refuri'] = node['anchorname'] or '#'
        return toc

    def get_toctree_for(self, docname: str, builder: 'Builder', collapse: bool, **kwargs: Any) -> Optional[Element]:
        """Return the global TOC nodetree."""
        doctree = self.env.get_doctree(self.env.config.root_doc)
        toctrees: List[Element] = []
        if 'includehidden' not in kwargs:
            kwargs['includehidden'] = True
        if 'maxdepth' not in kwargs or not kwargs['maxdepth']:
            kwargs['maxdepth'] = 0
        else:
            kwargs['maxdepth'] = int(kwargs['maxdepth'])
        kwargs['collapse'] = collapse
        for toctreenode in doctree.findall(addnodes.toctree):
            toctree = self.resolve(docname, builder, toctreenode, prune=True, **kwargs)
            if toctree:
                toctrees.append(toctree)
        if not toctrees:
            return None
        result = toctrees[0]
        for toctree in toctrees[1:]:
            result.extend(toctree.children)
        return result