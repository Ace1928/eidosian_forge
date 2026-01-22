import os
from botocore import xform_name
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.method import document_model_driven_method
from botocore.docs.utils import DocumentedShape
from botocore.utils import get_service_module_name
class PaginatorDocumenter:

    def __init__(self, client, service_paginator_model, root_docs_path):
        self._client = client
        self._client_class_name = self._client.__class__.__name__
        self._service_name = self._client.meta.service_model.service_name
        self._service_paginator_model = service_paginator_model
        self._root_docs_path = root_docs_path
        self._USER_GUIDE_LINK = 'https://boto3.amazonaws.com/v1/documentation/api/latest/guide/paginators.html'

    def document_paginators(self, section):
        """Documents the various paginators for a service

        param section: The section to write to.
        """
        section.style.h2('Paginators')
        self._add_overview(section)
        section.style.new_line()
        section.writeln('The available paginators are:')
        section.style.toctree()
        paginator_names = sorted(self._service_paginator_model._paginator_config)
        for paginator_name in paginator_names:
            section.style.tocitem(f'{self._service_name}/paginator/{paginator_name}')
            paginator_doc_structure = DocumentStructure(paginator_name, target='html')
            self._add_paginator(paginator_doc_structure, paginator_name)
            paginator_dir_path = os.path.join(self._root_docs_path, self._service_name, 'paginator')
            paginator_doc_structure.write_to_file(paginator_dir_path, paginator_name)

    def _add_paginator(self, section, paginator_name):
        breadcrumb_section = section.add_new_section('breadcrumb')
        breadcrumb_section.style.ref(self._client_class_name, f'../../{self._service_name}')
        breadcrumb_section.write(f' / Paginator / {paginator_name}')
        section.add_title_section(paginator_name)
        paginator_section = section.add_new_section(paginator_name)
        paginator_section.style.start_sphinx_py_class(class_name=f'{self._client_class_name}.Paginator.{paginator_name}')
        paginator_section.style.start_codeblock()
        paginator_section.style.new_line()
        paginator_section.write(f"paginator = client.get_paginator('{xform_name(paginator_name)}')")
        paginator_section.style.end_codeblock()
        paginator_section.style.new_line()
        paginator_config = self._service_paginator_model.get_paginator(paginator_name)
        document_paginate_method(section=paginator_section, paginator_name=paginator_name, event_emitter=self._client.meta.events, service_model=self._client.meta.service_model, paginator_config=paginator_config)

    def _add_overview(self, section):
        section.style.new_line()
        section.write('Paginators are available on a client instance via the ``get_paginator`` method. For more detailed instructions and examples on the usage of paginators, see the paginators ')
        section.style.external_link(title='user guide', link=self._USER_GUIDE_LINK)
        section.write('.')
        section.style.new_line()