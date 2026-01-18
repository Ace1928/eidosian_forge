import time
from tests.unit import unittest
from boto.datapipeline import layer1
def test_put_pipeline_definition(self):
    pipeline_id = self.create_pipeline('name3', 'unique_id3')
    self.connection.put_pipeline_definition(self.sample_pipeline_objects, pipeline_id)
    response = self.connection.get_pipeline_definition(pipeline_id)
    objects = response['pipelineObjects']
    self.assertEqual(len(objects), 3)
    self.assertEqual(objects[0]['id'], 'Default')
    self.assertEqual(objects[0]['name'], 'Default')
    self.assertEqual(objects[0]['fields'], [{'key': 'workerGroup', 'stringValue': 'MyworkerGroup'}])