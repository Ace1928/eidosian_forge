import copy
from parlai.core.teachers import ParlAIDialogTeacher, FbDialogTeacher
class Parley(object):
    """A single example for training
    Args:
        context: (str) the dialog context in which the response was given;
            - at minimum, this is the immediately previous utterance
            - at maximum, this is the entire conversation up to that point
            - external knowledge/memories should be in memories
        response: (str) response
        reward: (int) reward
        candidates: (list) of strings
        memories: (list) of strings in order (if it matters)
    """

    def __init__(self, context, response='', reward=0, candidates=None, memories=None, episode_done=False, **kwargs):
        if candidates is None:
            candidates = []
        if memories is None:
            memories = []
        self.context = context
        self.response = response if response is not None else ''
        self.reward = reward
        self.candidates = candidates
        self.memories = memories
        self.episode_done = bool(episode_done)

    def __repr__(self):
        return f'Parley({self.to_dict()})'

    def to_dict(self, include_empty=False):
        if include_empty:
            return {'context': self.context, 'response': self.response, 'reward': self.reward, 'candidates': self.candidates, 'memories': self.memories, 'episode_done': self.episode_done}
        else:
            pdict = {'context': self.context, 'response': self.response}
            if self.reward:
                pdict['reward'] = self.reward
            if self.candidates:
                pdict['candidates'] = self.candidates
            if self.memories:
                pdict['memories'] = self.memories
            if self.episode_done:
                pdict['episode_done'] = self.episode_done
            return pdict

    def to_parlai(self):
        string = f'context:{self.context}'
        string += f'\tresponse:{self.response}' if self.response else ''
        string += f'\treward:{self.reward}' if self.reward else ''
        string += f'\tcandidates:{'|'.join(self.candidates)}' if self.candidates else ''
        string += f'\tmemories:{'|'.join(self.candidates)}' if self.candidates else ''
        string += f'\tepisode_done:{self.episode_done}' if self.episode_done else ''
        return string.strip()

    def to_fb(self):
        pieces = [self.context, self.response, str(self.reward), '|'.join(self.candidates), '|'.join(self.memories)]
        return '\t'.join(pieces).strip()