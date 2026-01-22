import os
from ..exceptions import XMLSchemaException, XMLSchemaValueError
from ..names import XSD_NAMESPACE, WSDL_NAMESPACE, SOAP_NAMESPACE, \
from ..helpers import get_qname, local_name, get_extended_qname, get_prefixed_qname
from ..namespaces import NamespaceResourcesMap
from ..resources import fetch_resource
from ..documents import XmlDocument
from ..validators import XMLSchemaBase, XMLSchema10
def _parse_imports(self):
    namespace_imports = NamespaceResourcesMap(map(lambda x: (x.get('namespace', ''), x.get('location', '')), filter(lambda x: x.tag == WSDL_IMPORT, self.root)))
    for namespace, locations in namespace_imports.items():
        locations = [url for url in locations if url]
        try:
            locations.extend(self.locations[namespace])
        except KeyError:
            pass
        import_error = None
        for url in locations:
            try:
                self.import_namespace(namespace, url, self.base_url)
            except (OSError, IOError) as err:
                if import_error is None:
                    import_error = err
            except SyntaxError as err:
                msg = 'cannot import namespace %r: %s.' % (namespace, err)
                self.parse_error(msg)
            except XMLSchemaValueError as err:
                self.parse_error(err)
            else:
                break
        else:
            if import_error is not None:
                msg = 'import of namespace {!r} from {!r} failed: {}.'
                self.parse_error(msg.format(namespace, locations, str(import_error)))
            self.maps.imports[namespace] = None