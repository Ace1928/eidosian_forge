import numpy as np
def test_show_topics(self):
    topics = self.model.show_topics(formatted=False)
    for topic_no, topic in topics:
        self.assertTrue(isinstance(topic_no, int))
        self.assertTrue(isinstance(topic, list))
        for k, v in topic:
            self.assertTrue(isinstance(k, str))
            self.assertTrue(isinstance(v, (np.floating, float)))