import numpy as np
def test_show_topic(self):
    topic = self.model.show_topic(1)
    for k, v in topic:
        self.assertTrue(isinstance(k, str))
        self.assertTrue(isinstance(v, (np.floating, float)))