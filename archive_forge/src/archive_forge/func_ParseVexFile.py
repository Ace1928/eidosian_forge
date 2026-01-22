from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
import json
import re
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.container.images import util as gcr_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.core import log
from  googlecloudsdk.core.util.files import FileReader
def ParseVexFile(filename, image_uri, version_uri):
    """Reads a vex file and extracts notes.

  Args:
    filename: str, path to the vex file.
    image_uri: uri of the whole image
    version_uri: uri of a specific version

  Returns:
    A list of notes.

  Raises:
    ar_exceptions.InvalidInputValueError if user input is invalid.
  """
    ca_messages = apis.GetMessagesModule('containeranalysis', 'v1')
    try:
        with FileReader(filename) as file:
            vex = json.load(file)
    except ValueError:
        raise ar_exceptions.InvalidInputValueError('Reading json file has failed')
    _Validate(vex)
    name = ''
    namespace = ''
    document = vex.get('document')
    if document is not None:
        publisher = document.get('publisher')
        if publisher is not None:
            name = publisher.get('name')
            namespace = publisher.get('namespace')
    publisher = ca_messages.Publisher(name=name, publisherNamespace=namespace)
    generic_uri = version_uri if version_uri else image_uri
    productid_to_product_proto_map = {}
    for product_info in vex['product_tree']['branches']:
        artifact_uri = product_info['name']
        artifact_uri = RemoveHTTPS(artifact_uri)
        if image_uri != artifact_uri:
            continue
        product = product_info['product']
        product_id = product['product_id']
        generic_uri = 'https://{}'.format(generic_uri)
        product_proto = ca_messages.Product(name=product['name'], id=product_id, genericUri=generic_uri)
        productid_to_product_proto_map[product_id] = product_proto
    notes = []
    for vuln in vex['vulnerabilities']:
        for status in vuln['product_status']:
            for product_id in vuln['product_status'][status]:
                product = productid_to_product_proto_map.get(product_id)
                if product is None:
                    continue
                noteid, note = _MakeNote(vuln, status, product, publisher, document, ca_messages)
                if version_uri is None:
                    noteid = 'image-{}'.format(noteid)
                note = ca_messages.BatchCreateNotesRequest.NotesValue.AdditionalProperty(key=noteid, value=note)
                notes.append(note)
    return (notes, generic_uri)