import logging
import os
import re
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.docstringparser import DocStringParser
from botocore.docs.bcdoc.style import ReSTStyle
class DocumentStructure(ReSTDocument):

    def __init__(self, name, section_names=None, target='man', context=None):
        """Provides a Hierarichial structure to a ReSTDocument

        You can write to it similiar to as you can to a ReSTDocument but
        has an innate structure for more orginaztion and abstraction.

        :param name: The name of the document
        :param section_names: A list of sections to be included
            in the document.
        :param target: The target documentation of the Document structure
        :param context: A dictionary of data to store with the strucuture. These
            are only stored per section not the entire structure.
        """
        super().__init__(target=target)
        self._name = name
        self._structure = OrderedDict()
        self._path = [self._name]
        self._context = {}
        if context is not None:
            self._context = context
        if section_names is not None:
            self._generate_structure(section_names)

    @property
    def name(self):
        """The name of the document structure"""
        return self._name

    @property
    def path(self):
        """
        A list of where to find a particular document structure in the
        overlying document structure.
        """
        return self._path

    @path.setter
    def path(self, value):
        self._path = value

    @property
    def available_sections(self):
        return list(self._structure)

    @property
    def context(self):
        return self._context

    def _generate_structure(self, section_names):
        for section_name in section_names:
            self.add_new_section(section_name)

    def add_new_section(self, name, context=None):
        """Adds a new section to the current document structure

        This document structure will be considered a section to the
        current document structure but will in itself be an entirely
        new document structure that can be written to and have sections
        as well

        :param name: The name of the section.
        :param context: A dictionary of data to store with the strucuture. These
            are only stored per section not the entire structure.
        :rtype: DocumentStructure
        :returns: A new document structure to add to but lives as a section
            to the document structure it was instantiated from.
        """
        section = self.__class__(name=name, target=self.target, context=context)
        section.path = self.path + [name]
        section.style.indentation = self.style.indentation
        section.translation_map = self.translation_map
        section.hrefs = self.hrefs
        self._structure[name] = section
        return section

    def get_section(self, name):
        """Retrieve a section"""
        return self._structure[name]

    def delete_section(self, name):
        """Delete a section"""
        del self._structure[name]

    def flush_structure(self, docs_link=None):
        """Flushes a doc structure to a ReSTructed string

        The document is flushed out in a DFS style where sections and their
        subsections' values are added to the string as they are visited.
        """
        path_length = len(self.path)
        if path_length == 1:
            if self.hrefs:
                self.style.new_paragraph()
                for refname, link in self.hrefs.items():
                    self.style.link_target_definition(refname, link)
        elif path_length == SECTION_METHOD_PATH_DEPTH.get(self.path[1]):
            docs_link = None
        value = self.getvalue()
        for name, section in self._structure.items():
            match = DOCUMENTATION_LINK_REGEX.search(value.decode())
            docs_link = f'{match.group(0)}\n\n'.encode() if match else docs_link
            value += section.flush_structure(docs_link)
        line_count = len(value.splitlines())
        section_config = SECTION_LINE_LIMIT_CONFIG.get(self.name)
        aws_docs_link = docs_link.decode() if docs_link is not None else DEFAULT_AWS_DOCS_LINK
        if section_config and line_count > section_config['line_limit']:
            value = LARGE_SECTION_MESSAGE.format(section_config['name'], aws_docs_link).encode()
        return value

    def getvalue(self):
        return ''.join(self._writes).encode('utf-8')

    def remove_all_sections(self):
        self._structure = OrderedDict()

    def clear_text(self):
        self._writes = []

    def add_title_section(self, title):
        title_section = self.add_new_section('title')
        title_section.style.h1(title)
        return title_section

    def write_to_file(self, full_path, file_name):
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        sub_resource_file_path = os.path.join(full_path, f'{file_name}.rst')
        with open(sub_resource_file_path, 'wb') as f:
            f.write(self.flush_structure())