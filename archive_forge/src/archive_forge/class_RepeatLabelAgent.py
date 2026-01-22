import random
from parlai.core.agents import Agent
from parlai.core.message import Message
class RepeatLabelAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        group = argparser.add_argument_group('RepeatLabel Arguments')
        group.add_argument('--return_one_random_answer', type='bool', default=True, help='return one answer from the set of labels')
        group.add_argument('--cant_answer_percent', type=float, default=0, help='set value in range[0,1] to set chance of replying with special message')
        group.add_argument('--cant_answer_message', type=str, default="I don't know.", help='Message sent when the model cannot answer')

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.returnOneRandomAnswer = opt.get('return_one_random_answer', True)
        self.cantAnswerPercent = opt.get('cant_answer_percent', 0)
        self.cantAnswerMessage = opt.get('cant_answer_message', "I don't know.")
        self.id = 'RepeatLabelAgent'

    def act(self):
        obs = self.observation
        if obs is None:
            return {'text': 'Nothing to repeat yet.'}
        reply = {}
        reply['id'] = self.getID()
        labels = obs.get('labels', obs.get('eval_labels', None))
        if labels:
            if random.random() >= self.cantAnswerPercent:
                if self.returnOneRandomAnswer:
                    reply['text'] = labels[random.randrange(len(labels))]
                else:
                    reply['text'] = ', '.join(labels)
            else:
                reply['text'] = self.cantAnswerMessage
        else:
            reply['text'] = self.cantAnswerMessage
        reply['episode_done'] = False
        return Message(reply)