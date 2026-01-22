from parlai.core.teachers import FixedDialogTeacher
from . import tm_utils
import json
class WozDialogueTeacher(FixedDialogTeacher):
    """
    Teacher for spoken two-person dialogues with labels being responses for the previous
    statement.

    The data is traversed twice (doubled), once for modelling USER replies and once for
    modelling ASSISTANT replies.
    """

    def __init__(self, opt, shared=None):
        opt['fn'] = 'woz-dialogs.json'
        super().__init__(opt)
        if shared and 'convos' in shared:
            self.convos = shared['convos']
            self.episode_map = shared['episode_map']
            self.ep_cheat_sheet = shared['ep_cheat_sheet']
            self.num_ex = shared['num_ex']
        else:
            self.ep_cheat_sheet = {}
            self.episode_map = {}
            self.episode_map['U'] = {}
            self.episode_map['A'] = {}
            self.num_ex = 0
            data_path = tm_utils._path(opt)
            self._setup_data(data_path, opt)
        self.reset()

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('Corrupt-Example-Arguments')
        agent.add_argument('--exclude-invalid-data', type='bool', default=True, help='Whether to include corrupt examples in the data')

    def _setup_data(self, data_path, opt):
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            self.convos = json.load(data_file)
        convos_update = []
        for convo in self.convos:
            conversation, corrupted = tm_utils.smoothen_convo(convo, opt)
            if len(conversation) > 1 and (not corrupted):
                actual_ep_idx = len(self.ep_cheat_sheet)
                self.ep_cheat_sheet[actual_ep_idx] = tm_utils.gen_ep_cheatsheet(conversation)
                curr_cheatsheet = self.ep_cheat_sheet[len(self.ep_cheat_sheet) - 1]
                self.num_ex += curr_cheatsheet[tm_utils.USER_NUM_EX] + curr_cheatsheet[tm_utils.ASSIS_NUM_EX]
                if curr_cheatsheet[tm_utils.USER_NUM_EX] != 0:
                    u_idx = len(self.episode_map['U'])
                    self.episode_map['U'][u_idx] = actual_ep_idx
                if curr_cheatsheet[tm_utils.ASSIS_NUM_EX] != 0:
                    a_idx = len(self.episode_map['A'])
                    self.episode_map['A'][a_idx] = actual_ep_idx
                convos_update += [conversation]
        self.convos = convos_update

    def num_examples(self):
        return self.num_ex

    def num_episodes(self):
        return len(self.episode_map['U']) + len(self.episode_map['A'])

    def get(self, episode_idx, entry_idx):
        if episode_idx < len(self.episode_map['U']):
            true_idx = self.episode_map['U'][episode_idx]
            conversation = self.convos[true_idx]
            convo_cheat_sheet = self.ep_cheat_sheet[true_idx]
            first_entry_idx, last_entry_idx = (convo_cheat_sheet[tm_utils.FIRST_USER_IDX], convo_cheat_sheet[tm_utils.LAST_USER_IDX])
        else:
            episode_idx -= len(self.episode_map['U'])
            true_idx = self.episode_map['A'][episode_idx]
            conversation = self.convos[true_idx]
            convo_cheat_sheet = self.ep_cheat_sheet[true_idx]
            first_entry_idx, last_entry_idx = (convo_cheat_sheet[tm_utils.FIRST_ASSISTANT_IDX], convo_cheat_sheet[tm_utils.LAST_ASSISTANT_IDX])
        starts_at_odd = first_entry_idx % 2 != 0
        if starts_at_odd:
            predecessor = conversation[entry_idx * 2 + 1]['text']
            successor = conversation[entry_idx * 2 + 2]['text']
            ep_done = entry_idx * 2 + 1 == last_entry_idx
        else:
            predecessor = conversation[entry_idx * 2]['text']
            successor = conversation[entry_idx * 2 + 1]['text']
            ep_done = entry_idx * 2 == last_entry_idx
        action = {'id': self.id, 'text': predecessor, 'episode_done': ep_done, 'labels': [successor]}
        return action