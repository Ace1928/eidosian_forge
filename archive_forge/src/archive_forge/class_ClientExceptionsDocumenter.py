import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.example import ResponseExampleDocumenter
from botocore.docs.method import (
from botocore.docs.params import ResponseParamsDocumenter
from botocore.docs.sharedexample import document_shared_examples
from botocore.docs.utils import DocumentedShape, get_official_service_name
class ClientExceptionsDocumenter:
    _USER_GUIDE_LINK = 'https://boto3.amazonaws.com/v1/documentation/api/latest/guide/error-handling.html'
    _GENERIC_ERROR_SHAPE = DocumentedShape(name='Error', type_name='structure', documentation='Normalized access to common exception attributes.', members=OrderedDict([('Code', DocumentedShape(name='Code', type_name='string', documentation='An identifier specifying the exception type.')), ('Message', DocumentedShape(name='Message', type_name='string', documentation='A descriptive message explaining why the exception occured.'))]))

    def __init__(self, client, root_docs_path):
        self._client = client
        self._client_class_name = self._client.__class__.__name__
        self._service_name = self._client.meta.service_model.service_name
        self._root_docs_path = root_docs_path

    def document_exceptions(self, section):
        self._add_title(section)
        self._add_overview(section)
        self._add_exceptions_list(section)
        self._add_exception_classes()

    def _add_title(self, section):
        section.style.h2('Client Exceptions')

    def _add_overview(self, section):
        section.style.new_line()
        section.write('Client exceptions are available on a client instance via the ``exceptions`` property. For more detailed instructions and examples on the exact usage of client exceptions, see the error handling ')
        section.style.external_link(title='user guide', link=self._USER_GUIDE_LINK)
        section.write('.')
        section.style.new_line()

    def _exception_class_name(self, shape):
        return f'{self._client_class_name}.Client.exceptions.{shape.name}'

    def _add_exceptions_list(self, section):
        error_shapes = self._client.meta.service_model.error_shapes
        if not error_shapes:
            section.style.new_line()
            section.write('This client has no modeled exception classes.')
            section.style.new_line()
            return
        section.style.new_line()
        section.writeln('The available client exceptions are:')
        section.style.toctree()
        for shape in error_shapes:
            section.style.tocitem(f'{self._service_name}/client/exceptions/{shape.name}')

    def _add_exception_classes(self):
        for shape in self._client.meta.service_model.error_shapes:
            exception_doc_structure = DocumentStructure(shape.name, target='html')
            self._add_exception_class(exception_doc_structure, shape)
            exception_dir_path = os.path.join(self._root_docs_path, self._service_name, 'client', 'exceptions')
            exception_doc_structure.write_to_file(exception_dir_path, shape.name)

    def _add_exception_class(self, section, shape):
        breadcrumb_section = section.add_new_section('breadcrumb')
        breadcrumb_section.style.ref(self._client_class_name, f'../../../{self._service_name}')
        breadcrumb_section.write(f' / Client / exceptions / {shape.name}')
        section.add_title_section(shape.name)
        class_section = section.add_new_section(shape.name)
        class_name = self._exception_class_name(shape)
        class_section.style.start_sphinx_py_class(class_name=class_name)
        self._add_top_level_documentation(class_section, shape)
        self._add_exception_catch_example(class_section, shape)
        self._add_response_attr(class_section, shape)
        class_section.style.end_sphinx_py_class()

    def _add_top_level_documentation(self, section, shape):
        if shape.documentation:
            section.style.new_line()
            section.include_doc_string(shape.documentation)
            section.style.new_line()

    def _add_exception_catch_example(self, section, shape):
        section.style.new_line()
        section.style.bold('Example')
        section.style.new_paragraph()
        section.style.start_codeblock()
        section.write('try:')
        section.style.indent()
        section.style.new_line()
        section.write('...')
        section.style.dedent()
        section.style.new_line()
        section.write('except client.exceptions.%s as e:' % shape.name)
        section.style.indent()
        section.style.new_line()
        section.write('print(e.response)')
        section.style.dedent()
        section.style.end_codeblock()

    def _add_response_attr(self, section, shape):
        response_section = section.add_new_section('response')
        response_section.style.start_sphinx_py_attr('response')
        self._add_response_attr_description(response_section)
        self._add_response_example(response_section, shape)
        self._add_response_params(response_section, shape)
        response_section.style.end_sphinx_py_attr()

    def _add_response_attr_description(self, section):
        section.style.new_line()
        section.include_doc_string('The parsed error response. All exceptions have a top level ``Error`` key that provides normalized access to common exception atrributes. All other keys are specific to this service or exception class.')
        section.style.new_line()

    def _add_response_example(self, section, shape):
        example_section = section.add_new_section('syntax')
        example_section.style.new_line()
        example_section.style.bold('Syntax')
        example_section.style.new_paragraph()
        documenter = ResponseExampleDocumenter(service_name=self._service_name, operation_name=None, event_emitter=self._client.meta.events)
        documenter.document_example(example_section, shape, include=[self._GENERIC_ERROR_SHAPE])

    def _add_response_params(self, section, shape):
        params_section = section.add_new_section('Structure')
        params_section.style.new_line()
        params_section.style.bold('Structure')
        params_section.style.new_paragraph()
        documenter = ResponseParamsDocumenter(service_name=self._service_name, operation_name=None, event_emitter=self._client.meta.events)
        documenter.document_params(params_section, shape, include=[self._GENERIC_ERROR_SHAPE])