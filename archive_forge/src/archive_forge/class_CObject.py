import lxml
import os
import os.path as op
import sys
import re
import shutil
import tempfile
import zipfile
import codecs
from fnmatch import fnmatch
from itertools import islice
from lxml import etree
from pathlib import Path
from .uriutil import join_uri, translate_uri, uri_segment
from .uriutil import uri_last, uri_nextlast
from .uriutil import uri_parent, uri_grandparent
from .uriutil import uri_shape
from .uriutil import file_path
from .jsonutil import JsonTable, get_selection
from .pathutil import find_files, ensure_dir_exists
from .attributes import EAttrs
from .search import rpn_contraints, query_from_xml
from .errors import is_xnat_error, parse_put_error_message
from .errors import DataError, ProgrammingError, catch_error
from .provenance import Provenance
from .pipelines import Pipelines
from . import schema
from . import httputil
from . import downloadutils
from . import derivatives
import types
import pkgutil
import inspect
from urllib.parse import quote, unquote
class CObject(object):
    """ Generic Object for a collection resource.

        A collection resource is a list of element resources. There is
        however several ways to obtain such a list:

            - a collection URI e.g. /REST/projects
            - a list of element URIs
            - a list of collections
               e.g. /REST/projects/ONE/subjects **AND**
               /REST/projects/TWO/subjects
            - a list of element objects
            - a list a collection objects

        Collections objects built in different ways share the same behavior:

            - they behave as iterators, which enables a lazy access to
              the data
            - they always yield EObjects
            - they can be nested with any other collection

        Examples
        --------
        No access to the data:
            >>> interface.select.projects()
            <Collection Object> 173667084

        Lazy access to the data:
            >>> for project in interface.select.projects():
            >>>     print project

        Nesting:
            >>> for subject in interface.select.projects().subjects():
            >>>     print subject
    """

    def __init__(self, cbase, interface, pattern='*', nested=None, id_header='ID', columns=[], filters={}):
        """
            Parameters
            ----------
            cbase: string | list | CObject
                Object from which the collection is built.
            interface: :class:`Interface`
                Main interface reference.
            pattern: string
                Only resource element whose ID match the pattern are
                returned.
            nested: None | string
                Parameter used to nest collections.
            id_header: ID | label
                Defines whether the element label or ID is returned as the
                identifier.
            columns: list
                Defines additional columns to be returned.
            filters: dict
                Defines additional filters for the query, typically options
                for the query string.
        """
        self._intf = interface
        self._cbase = cbase
        self._id_header = id_header
        self._pattern = pattern
        self._columns = columns
        self._filters = filters
        self._nested = nested
        if isinstance(cbase, str):
            self._ctype = 'cobjectcuri'
        elif isinstance(cbase, CObject):
            self._ctype = 'cobjectcobject'
        elif isinstance(cbase, list) and cbase:
            if isinstance(cbase[0], str):
                self._ctype = 'cobjecteuris'
            if isinstance(cbase[0], EObject):
                self._ctype = 'cobjecteobjects'
            if isinstance(cbase[0], CObject):
                self._ctype = 'cobjectcobjects'
        elif isinstance(cbase, list) and (not cbase):
            self._ctype = 'cobjectempty'
        else:
            raise Exception('Invalid collection accessor type: %s' % cbase)

    def __repr__(self):
        return '<Collection Object> %s' % id(self)

    def _call(self, columns):
        try:
            uri = translate_uri(self._cbase)
            uri = quote(uri)
            columns += ['xsiType']
            query_string = '?format=json&columns=%s' % ','.join(columns)
            if self._filters:
                query_string += '&' + '&'.join(('%s=%s' % (item[0], item[1]) if isinstance(item[1], str) else '%s=%s' % (item[0], ','.join([val for val in item[1]])) for item in self._filters.items()))
            if DEBUG:
                print(uri + query_string)
            jtable = self._intf._get_json(uri + query_string)
            _type = uri.split('/')[-1]
            self._learn_from_table(_type, jtable, None)
            return jtable
        except Exception as e:
            if DEBUG:
                raise e
            return []

    def _learn_from_table(self, _type, jtable, reqcache):
        request_knowledge = {}
        for element in jtable:
            xsitype = element.get('xsiType')
            if xsitype:
                uri = element.get('URI').split(self._intf._get_entry_point(), 1)[1]
                uri = uri.replace(uri.split('/')[-2], _type)
                shape = uri_shape(uri)
                request_knowledge[shape] = xsitype
        self._intf._struct.update(request_knowledge)

    def __iter__(self):
        if self._ctype == 'cobjectcuri':
            if self._id_header == 'ID':
                id_header = schema.json[uri_last(self._cbase)][0]
            elif self._id_header == 'label':
                id_header = schema.json[uri_last(self._cbase)][1]
            else:
                id_header = self._id_header
            for res in self._call([id_header] + self._columns):
                try:
                    eid = unquote(res[id_header])
                    if fnmatch(eid, self._pattern):
                        klass_name = uri_last(self._cbase).rstrip('s').title()
                        Klass = globals().get(klass_name, self._intf.__class__)
                        eobj = Klass(join_uri(self._cbase, eid), self._intf)
                        if self._nested is None:
                            self._run_callback(self, eobj)
                            yield eobj
                        else:
                            Klass = globals().get(self._nested.title(), self._intf.__class__)
                            for subeobj in Klass(cbase=join_uri(eobj._uri, self._nested), interface=self._intf, pattern=self._pattern, id_header=self._id_header, columns=self._columns):
                                try:
                                    self._run_callback(self, subeobj)
                                    yield subeobj
                                except RuntimeError:
                                    pass
                except KeyboardInterrupt:
                    self._intf._connect()
                    raise StopIteration
        elif self._ctype == 'cobjecteuris':
            for uri in self._cbase:
                try:
                    title = uri_nextlast(uri).rstrip('s').title()
                    Klass = globals().get(title, self._intf.__class__)
                    eobj = Klass(uri, self._intf)
                    if self._nested is None:
                        self._run_callback(self, eobj)
                        yield eobj
                    else:
                        Klass = globals().get(self._nested.title(), self._intf.__class__)
                        for subeobj in Klass(cbase=join_uri(eobj._uri, self._nested), interface=self._intf, pattern=self._pattern, id_header=self._id_header, columns=self._columns):
                            try:
                                self._run_callback(self, subeobj)
                                yield subeobj
                            except RuntimeError:
                                pass
                except KeyboardInterrupt:
                    self._intf._connect()
                    raise StopIteration
        elif self._ctype == 'cobjecteobjects':
            for eobj in self._cbase:
                try:
                    if self._nested is None:
                        self._run_callback(self, eobj)
                        yield eobj
                    else:
                        Klass = globals().get(self._nested.rstrip('s').title(), self._intf.__class__)
                        for subeobj in Klass(cbase=join_uri(eobj._uri, self._nested), interface=self._intf, pattern=self._pattern, id_header=self._id_header, columns=self._columns):
                            try:
                                self._run_callback(self, subeobj)
                                yield subeobj
                            except RuntimeError:
                                pass
                except KeyboardInterrupt:
                    self._intf._connect()
                    raise StopIteration
        elif self._ctype == 'cobjectcobject':
            for eobj in self._cbase:
                try:
                    if self._nested is None:
                        self._run_callback(self, eobj)
                        yield eobj
                    else:
                        Klass = globals().get(self._nested.title(), self._intf.__class__)
                        for subeobj in Klass(cbase=join_uri(eobj._uri, self._nested), interface=self._intf, pattern=self._pattern, id_header=self._id_header, columns=self._columns):
                            try:
                                self._run_callback(self, eobj)
                                yield subeobj
                            except RuntimeError:
                                pass
                except KeyboardInterrupt:
                    self._intf._connect()
                    raise StopIteration
        elif self._ctype == 'cobjectcobjects':
            for cobj in self._cbase:
                try:
                    for eobj in cobj:
                        if self._nested is None:
                            self._run_callback(self, eobj)
                            yield eobj
                        else:
                            Klass = globals().get(cobj._nested.title(), self._intf.__class__)
                            for subeobj in Klass(cbase=join_uri(eobj._uri, cobj._nested), interface=cobj._intf, pattern=cobj._pattern, id_header=cobj._id_header, columns=cobj._columns):
                                try:
                                    self._run_callback(self, eobj)
                                    yield subeobj
                                except RuntimeError:
                                    pass
                except KeyboardInterrupt:
                    self._intf._connect()
                    raise StopIteration
        elif self._ctype == 'cobjectempty':
            for empty in []:
                yield empty

    def _run_callback(self, cobj, eobj):
        if self._intf._callback is not None:
            self._intf._callback(cobj, eobj)

    def first(self):
        """ Returns the first element of the collection.
        """
        for eobj in self:
            return eobj
    fetchone = first

    def __getitem__(self, k):
        """ Use itertools.islice() to support indexed access and slicing.
        """
        if isinstance(k, slice):
            return islice(self, k.start, k.stop, k.step)
        else:
            return next(islice(self, k, k + 1))

    def get(self, *args):
        """ Returns every element.

            .. warning::
                If a collection needs to issue thousands of queries it may
                be better to access the resources within a `for-loop`.

            Parameters
            ----------
            args: strings
                - Specify the information to return for the elements
                  within ID, label and Object.
                - Any combination of ID, label and obj is valid, if
                  more than one is given, a list of tuple is returned
                  instead of a list.
        """
        if not args:
            return [unquote(uri_last(eobj._uri)) for eobj in self]
        else:
            entries = []
            for eobj in self:
                entry = ()
                for arg in args:
                    if arg == 'id':
                        self._id_header = 'ID'
                        entry += (unquote(uri_last(eobj._uri)),)
                    elif arg == 'label':
                        self._id_header = 'label'
                        entry += (unquote(uri_last(eobj._uri)),)
                    else:
                        entry += (eobj,)
                entries.append(entry)
            if len(args) != 1:
                return entries
            else:
                return [e[0] for e in entries]
    fetchall = get

    def __nonzero__(self):
        try:
            self.__iter__().next()
        except StopIteration:
            return False
        else:
            return True

    def tag(self, name):
        """ Tag the collection.
        """
        tag = self._intf.manage.tags.get(name)
        if not tag.exists():
            tag.create()
        tag.reference_many([eobj._uri for eobj in self])
        return tag

    def untag(self, name):
        """ Remove the tag from the collection.
        """
        tag = self._intf.manage.tags.get(name)
        tag.dereference_many([eobj._uri for eobj in self])
        if not tag.references().get():
            tag.delete()

    def where(self, constraints=None, template=None, query=None):
        """ Only the element objects whose subject that are matching the
            constraints will be returned. It means that it is not possible
            to use this method on an element that is not linked to a
            subject, such as a project.

            Examples
            --------
            The ``where`` clause should be on the first select:
                >>> for experiment in interface.select('//experiments'
                         ).where([('atest/FIELD', '=', 'value'), 'AND']):
                >>>      print experiment

            Do **NOT** do this:
                >>> for experiment in interface.select('//experiments'):
                        for assessor in experiment.assessors(
                            ).where([('atest/FIELD', '=', 'value'), 'AND']):
                >>>         print assessor

            Or this:
                >>> for project in interface.select('//projects'
                        ).where([('atest/FIELD', '=', 'value'), 'AND']):
                >>>     print project

            See Also
            --------
            :func:`search.Search`
        """
        if isinstance(constraints, str):
            constraints = rpn_contraints(constraints)
        elif isinstance(template, tuple):
            tmp_bundle = self._intf.manage.search.get_template(template[0], True)
            tmp_bundle = tmp_bundle % template[1]
            constraints = query_from_xml(tmp_bundle)['constraints']
        elif isinstance(query, str):
            tmp_bundle = self._intf.manage.search.get(query, 'xml')
            constraints = query_from_xml(tmp_bundle)['constraints']
        elif isinstance(constraints, list):
            pass
        else:
            raise ProgrammingError('One in [contraints, template and query] parameters must be correctly set.')
        results = query_with(interface=self._intf, join_field='xnat:subjectData/SUBJECT_ID', common_field='SUBJECT_ID', return_values=['xnat:subjectData/PROJECT', 'xnat:subjectData/SUBJECT_ID'], _filter=constraints)
        searchpop = ['%s/projects/' % self._intf._get_entry_point() + '%(project)s/subjects/%(subject_id)s' % res for res in results]
        cobj = self
        while cobj:
            first = cobj.first()
            if not first:
                break
            if uri_nextlast(first._uri) == 'subjects':
                break
            else:
                cobj = getattr(cobj, '_cbase')
        backup_header = cobj._id_header
        if cobj._pattern != '*':
            cobj._id_header = 'ID'
            poi = set(searchpop).intersection([eobj._uri for eobj in cobj])
        else:
            poi = searchpop
        cobj._cbase = list(poi)
        cobj._ctype = 'cobjecteuris'
        cobj._nested = None
        cobj._id_header = backup_header
        return self