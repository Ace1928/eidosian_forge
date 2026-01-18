import numpy as np
def test_print_topics(self):
    topics = self.model.print_topics()
    for topic_no, topic in topics:
        self.assertTrue(isinstance(topic_no, int))
        self.assertTrue(isinstance(topic, str))