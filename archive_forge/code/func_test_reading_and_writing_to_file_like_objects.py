import unittest
from prov.model import *
from prov.dot import prov_to_dot
from prov.serializers import Registry
from prov.tests.examples import primer_example, primer_example_alternate
def test_reading_and_writing_to_file_like_objects(self):
    """
        Tests reading and writing to and from file like objects.
        """
    document = ProvDocument()
    document.entity(EX2_NS['test'])
    objects = [io.BytesIO, io.StringIO]
    Registry.load_serializers()
    formats = Registry.serializers.keys()
    for obj in objects:
        for format in formats:
            try:
                buf = obj()
                document.serialize(destination=buf, format=format)
                buf.seek(0, 0)
                new_document = ProvDocument.deserialize(source=buf, format=format)
                self.assertEqual(document, new_document)
            except NotImplementedError:
                pass
            finally:
                buf.close()