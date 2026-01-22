import re
from docutils import nodes, utils
from docutils.transforms import TransformError, Transform
class DocInfo(Transform):
    """
    This transform is specific to the reStructuredText_ markup syntax;
    see "Bibliographic Fields" in the `reStructuredText Markup
    Specification`_ for a high-level description. This transform
    should be run *after* the `DocTitle` transform.

    Given a field list as the first non-comment element after the
    document title and subtitle (if present), registered bibliographic
    field names are transformed to the corresponding DTD elements,
    becoming child elements of the "docinfo" element (except for a
    dedication and/or an abstract, which become "topic" elements after
    "docinfo").

    For example, given this document fragment after parsing::

        <document>
            <title>
                Document Title
            <field_list>
                <field>
                    <field_name>
                        Author
                    <field_body>
                        <paragraph>
                            A. Name
                <field>
                    <field_name>
                        Status
                    <field_body>
                        <paragraph>
                            $RCSfile$
            ...

    After running the bibliographic field list transform, the
    resulting document tree would look like this::

        <document>
            <title>
                Document Title
            <docinfo>
                <author>
                    A. Name
                <status>
                    frontmatter.py
            ...

    The "Status" field contained an expanded RCS keyword, which is
    normally (but optionally) cleaned up by the transform. The sole
    contents of the field body must be a paragraph containing an
    expanded RCS keyword of the form "$keyword: expansion text $". Any
    RCS keyword can be processed in any bibliographic field. The
    dollar signs and leading RCS keyword name are removed. Extra
    processing is done for the following RCS keywords:

    - "RCSfile" expands to the name of the file in the RCS or CVS
      repository, which is the name of the source file with a ",v"
      suffix appended. The transform will remove the ",v" suffix.

    - "Date" expands to the format "YYYY/MM/DD hh:mm:ss" (in the UTC
      time zone). The RCS Keywords transform will extract just the
      date itself and transform it to an ISO 8601 format date, as in
      "2000-12-31".

      (Since the source file for this text is itself stored under CVS,
      we can't show an example of the "Date" RCS keyword because we
      can't prevent any RCS keywords used in this explanation from
      being expanded. Only the "RCSfile" keyword is stable; its
      expansion text changes only if the file name changes.)

    .. _reStructuredText: http://docutils.sf.net/rst.html
    .. _reStructuredText Markup Specification:
       http://docutils.sf.net/docs/ref/rst/restructuredtext.html
    """
    default_priority = 340
    biblio_nodes = {'author': nodes.author, 'authors': nodes.authors, 'organization': nodes.organization, 'address': nodes.address, 'contact': nodes.contact, 'version': nodes.version, 'revision': nodes.revision, 'status': nodes.status, 'date': nodes.date, 'copyright': nodes.copyright, 'dedication': nodes.topic, 'abstract': nodes.topic}
    'Canonical field name (lowcased) to node class name mapping for\n    bibliographic fields (field_list).'

    def apply(self):
        if not getattr(self.document.settings, 'docinfo_xform', 1):
            return
        document = self.document
        index = document.first_child_not_matching_class(nodes.PreBibliographic)
        if index is None:
            return
        candidate = document[index]
        if isinstance(candidate, nodes.field_list):
            biblioindex = document.first_child_not_matching_class((nodes.Titular, nodes.Decorative))
            nodelist = self.extract_bibliographic(candidate)
            del document[index]
            document[biblioindex:biblioindex] = nodelist

    def extract_bibliographic(self, field_list):
        docinfo = nodes.docinfo()
        bibliofields = self.language.bibliographic_fields
        labels = self.language.labels
        topics = {'dedication': None, 'abstract': None}
        for field in field_list:
            try:
                name = field[0][0].astext()
                normedname = nodes.fully_normalize_name(name)
                if not (len(field) == 2 and normedname in bibliofields and self.check_empty_biblio_field(field, name)):
                    raise TransformError
                canonical = bibliofields[normedname]
                biblioclass = self.biblio_nodes[canonical]
                if issubclass(biblioclass, nodes.TextElement):
                    if not self.check_compound_biblio_field(field, name):
                        raise TransformError
                    utils.clean_rcs_keywords(field[1][0], self.rcs_keyword_substitutions)
                    docinfo.append(biblioclass('', '', *field[1][0]))
                elif issubclass(biblioclass, nodes.authors):
                    self.extract_authors(field, name, docinfo)
                elif issubclass(biblioclass, nodes.topic):
                    if topics[canonical]:
                        field[-1] += self.document.reporter.warning('There can only be one "%s" field.' % name, base_node=field)
                        raise TransformError
                    title = nodes.title(name, labels[canonical])
                    title[0].rawsource = labels[canonical]
                    topics[canonical] = biblioclass('', title, *field[1].children, classes=[canonical])
                else:
                    docinfo.append(biblioclass('', *field[1].children))
            except TransformError:
                if len(field[-1]) == 1 and isinstance(field[-1][0], nodes.paragraph):
                    utils.clean_rcs_keywords(field[-1][0], self.rcs_keyword_substitutions)
                classvalue = nodes.make_id(normedname)
                if classvalue:
                    field['classes'].append(classvalue)
                docinfo.append(field)
        nodelist = []
        if len(docinfo) != 0:
            nodelist.append(docinfo)
        for name in ('dedication', 'abstract'):
            if topics[name]:
                nodelist.append(topics[name])
        return nodelist

    def check_empty_biblio_field(self, field, name):
        if len(field[-1]) < 1:
            field[-1] += self.document.reporter.warning('Cannot extract empty bibliographic field "%s".' % name, base_node=field)
            return None
        return 1

    def check_compound_biblio_field(self, field, name):
        if len(field[-1]) > 1:
            field[-1] += self.document.reporter.warning('Cannot extract compound bibliographic field "%s".' % name, base_node=field)
            return None
        if not isinstance(field[-1][0], nodes.paragraph):
            field[-1] += self.document.reporter.warning('Cannot extract bibliographic field "%s" containing anything other than a single paragraph.' % name, base_node=field)
            return None
        return 1
    rcs_keyword_substitutions = [(re.compile('\\$Date: (\\d\\d\\d\\d)[-/](\\d\\d)[-/](\\d\\d)[ T][\\d:]+[^$]* \\$', re.IGNORECASE), '\\1-\\2-\\3'), (re.compile('\\$RCSfile: (.+),v \\$', re.IGNORECASE), '\\1'), (re.compile('\\$[a-zA-Z]+: (.+) \\$'), '\\1')]

    def extract_authors(self, field, name, docinfo):
        try:
            if len(field[1]) == 1:
                if isinstance(field[1][0], nodes.paragraph):
                    authors = self.authors_from_one_paragraph(field)
                elif isinstance(field[1][0], nodes.bullet_list):
                    authors = self.authors_from_bullet_list(field)
                else:
                    raise TransformError
            else:
                authors = self.authors_from_paragraphs(field)
            authornodes = [nodes.author('', '', *author) for author in authors if author]
            if len(authornodes) >= 1:
                docinfo.append(nodes.authors('', *authornodes))
            else:
                raise TransformError
        except TransformError:
            field[-1] += self.document.reporter.warning('Bibliographic field "%s" incompatible with extraction: it must contain either a single paragraph (with authors separated by one of "%s"), multiple paragraphs (one per author), or a bullet list with one paragraph (one author) per item.' % (name, ''.join(self.language.author_separators)), base_node=field)
            raise

    def authors_from_one_paragraph(self, field):
        """Return list of Text nodes for ";"- or ","-separated authornames."""
        text = ''.join((str(node) for node in field[1].traverse(nodes.Text)))
        if not text:
            raise TransformError
        for authorsep in self.language.author_separators:
            pattern = '(?<!\x00)%s' % authorsep
            authornames = re.split(pattern, text)
            if len(authornames) > 1:
                break
        authornames = (name.strip() for name in authornames)
        authors = [[nodes.Text(name, utils.unescape(name, True))] for name in authornames if name]
        return authors

    def authors_from_bullet_list(self, field):
        authors = []
        for item in field[1][0]:
            if isinstance(item, nodes.comment):
                continue
            if len(item) != 1 or not isinstance(item[0], nodes.paragraph):
                raise TransformError
            authors.append(item[0].children)
        if not authors:
            raise TransformError
        return authors

    def authors_from_paragraphs(self, field):
        for item in field[1]:
            if not isinstance(item, (nodes.paragraph, nodes.comment)):
                raise TransformError
        authors = [item.children for item in field[1] if not isinstance(item, nodes.comment)]
        return authors