from parlai.core.agents import create_agent_from_shared
from parlai.mturk.core.legacy_2018.agents import TIMEOUT_MESSAGE
from parlai.core.worlds import validate, MultiAgentDialogWorld
from parlai.mturk.core.legacy_2018.worlds import MTurkOnboardWorld
from parlai.core.message import Message
from parlai.utils.strings import normalize_reply
from joblib import Parallel, delayed
import numpy as np
import os
import json
import random
import time
import torch
import copy
class ControllableDialogEval(MultiAgentDialogWorld):

    def __init__(self, opt, agents=None, shared=None, num_turns=6, max_resp_time=120, model_agent_opt=None, world_tag='', agent_timeout_shutdown=120, model_config=None):
        self.opt = opt
        self.turn_idx = 0
        self.n_turn = num_turns
        self.chat_done = False
        self.other_first = random.choice([True, False])
        self.model_config = model_config
        self.start_time = time.time()
        self.dialog = []
        self.dialog_list = []
        self.engagingness_scores = []
        self.interestingness_scores = []
        self.listening_scores = []
        self.consistency_scores = []
        self.inquisitiveness_scores = []
        self.humanness_scores = []
        self.repetitiveness_scores = []
        self.fluency_scores = []
        self.persona_scores = []
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.world_tag = world_tag
        super().__init__(opt, agents, shared)
        if model_agent_opt is not None:
            self.model_agent = create_agent_from_shared(model_agent_opt)
        else:
            self.model_agent = None
        self.max_resp_time = max_resp_time
        self.agent_timeout_shutdown = agent_timeout_shutdown
        self.bot_seen_persona = False
        self.personas = [ag.personas for ag in self.agents]
        if self.model_agent is not None:
            self.eval_agent = self.agents[0]
            self.model_personas = self.agents[0].model_personas
            self.model_persona_text = '\n'.join(['your persona: ' + pers for pers in self.model_personas])
        else:
            self.model_personas = None
            for idx in range(len(self.agents)):
                if self.agents[idx].id == 'PERSON_1':
                    self.eval_agent = self.agents[idx]
                    self.other_agent = self.agents[idx - 1]
                    break

    def get_control_msg(self):
        return {'id': 'SYSTEM', 'episode_done': False}

    def get_human_agent_act(self, agent):
        act = agent.act(timeout=self.max_resp_time)
        while self.is_msg_tooshortlong(act, agent):
            act = agent.act(timeout=self.max_resp_time)
        return act

    def format_personachat_text(self, text):
        new_text = text.lower()
        switch_list = [("we're", 'were'), ("let's", 'lets'), ("it's", 'its'), ("who's", 'whos'), ("you're", 'youre'), ("you've", 'youve'), ("he'd", 'hed'), ("he'll", 'hell')]
        for tup in switch_list:
            new_text = new_text.replace(tup[0], tup[1])
        return new_text

    def get_bot_observation(self):
        pass

    def parley(self):
        self.turn_idx += 1
        print(self.world_tag + ' is at turn {}...'.format(self.turn_idx))
        'If at first turn, we need to give each agent their persona'
        if self.turn_idx == 1:
            for idx, agent in enumerate(self.agents):
                persona_text = ''
                for s in self.personas[idx]:
                    persona_text += '<b><span style="color:blue">{}\n</span></b>'.format(s.strip())
                control_msg = self.get_control_msg()
                control_msg['persona_text'] = persona_text
                control_msg['text'] = self.get_instruction(tag='start', agent_id=agent.id)
                agent.observe(validate(control_msg))
                if idx == 0:
                    time.sleep(3)
        'If we get to the min turns, inform turker that they can end if they\n        want.\n        '
        if self.turn_idx == self.n_turn + 1:
            for idx, agent in enumerate(self.agents):
                control_msg = self.get_control_msg()
                control_msg['text'] = self.get_instruction(idx, tag='exceed_min_turns')
                control_msg['exceed_min_turns'] = True
                agent.observe(validate(control_msg))
        'Otherwise, we proceed accordingly.'
        if self.other_first and self.turn_idx == 1:
            if self.model_agent is not None:
                persona_act = {'text': '\n'.join([self.model_persona_text, '__SILENCE__']), 'episode_done': False}
                self.model_agent.observe(persona_act)
                self.bot_seen_persona = True
                model_act = copy.deepcopy(self.model_agent.act())
                model_act.force_set('text', normalize_reply(model_act['text']))
                model_act.force_set('id', 'PERSON_2')
                self.dialog.append((1, model_act.get('text')))
                _random_delay()
                self.eval_agent.observe(_strip_tensors(model_act))
            else:
                act = self.get_human_agent_act(self.other_agent)
                timeout = self.check_timeout(act)
                if timeout:
                    control_msg = self.get_control_msg()
                    control_msg['text'] = UNEXPECTED_DISCONNECTION_MSG
                    self.eval_agent.observe(validate(control_msg))
                    return
                else:
                    self.dialog.append((1, act.get('text')))
                    act = copy.deepcopy(act)
                    act.force_set('text', normalize_reply(act['text']))
                    self.eval_agent.observe(act)
        act = Message(self.get_human_agent_act(self.eval_agent))
        timeout = self.check_timeout(act)
        if timeout:
            if self.model_agent is None:
                control_msg = self.get_control_msg()
                control_msg['text'] = UNEXPECTED_DISCONNECTION_MSG
                self.other_agent.observe(validate(control_msg))
            return
        if act['episode_done']:
            if self.turn_idx >= self.n_turn:
                if not self.other_first:
                    self.dialog_list = ['\n'.join([self.dialog[i][1], self.dialog[i + 1][1]]) for i in range(0, len(self.dialog), 2)]
                else:
                    self.dialog_list = [' \n' + self.dialog[0][1]] + ['\n'.join([self.dialog[i][1], self.dialog[i + 1][1]]) for i in range(1, len(self.dialog) - 1, 2)]
                self.parallel_eval_mode()
                self.chat_done = True
                for ag in self.agents:
                    control_msg = self.get_control_msg()
                    control_msg['text'] = CHAT_ENDED_MSG
                    ag.observe(validate(control_msg))
            return
        self.dialog.append((0, act['text']))
        if not self.bot_seen_persona and self.model_agent is not None:
            act.force_set('text', '\n'.join([self.model_persona_text, act['text']]))
            self.bot_seen_persona = True
        if self.model_agent is not None:
            self.model_agent.observe(act)
        else:
            act = copy.deepcopy(act)
            act.force_set('text', normalize_reply(act['text']))
            self.other_agent.observe(act)
        if not self.other_first or self.turn_idx < self.n_turn:
            if self.model_agent is not None:
                _random_delay()
                act = _strip_tensors(copy.deepcopy(self.model_agent.act()))
                act.force_set('text', normalize_reply(act['text']))
                act.force_set('id', 'PERSON_2')
            else:
                act = self.get_human_agent_act(self.other_agent)
                timeout = self.check_timeout(act)
                if timeout:
                    control_msg = self.get_control_msg()
                    control_msg['text'] = UNEXPECTED_DISCONNECTION_MSG
                    self.eval_agent.observe(validate(control_msg))
                    return
            self.dialog.append((1, act.get('text')))
            act = copy.deepcopy(act)
            act.force_set('text', normalize_reply(act['text']))
            self.eval_agent.observe(act)

    def _evaluate_characteristic(self, question, choices, addto):
        control_msg = self.get_control_msg()
        control_msg['text'] = question
        control_msg['button_choices'] = '</ROUND>'.join(choices)
        self.eval_agent.observe(validate(control_msg))
        act = self.eval_agent.act(timeout=self.max_resp_time)
        timeout = self.check_timeout(act)
        if timeout:
            return False
        act_choice = choices.index(act.get('text'))
        addto.append(act_choice)
        return True

    def evaluate_engagingness(self):
        control_msg = self.get_control_msg()
        msg_rng = len(ENGAGINGNESS_MSGS)
        for i in range(msg_rng):
            control_msg['text'] = ENGAGINGNESS_MSGS[i]
            control_msg['button_choices'] = '</ROUND>'.join(ENGAGINGNESS_CHOICES)
            self.eval_agent.observe(validate(control_msg))
            act = self.eval_agent.act(timeout=self.max_resp_time)
            timeout = self.check_timeout(act)
            if timeout:
                return False
            act_choice = ENGAGINGNESS_CHOICES.index(act.get('text'))
            self.engagingness_scores.append(act_choice)
        return True

    def evaluate_interestingness(self):
        return self._evaluate_characteristic(INTERESTINGNESS_MSGS[0], INTERESTINGNESS_CHOICES, self.interestingness_scores)

    def evaluate_listening(self):
        return self._evaluate_characteristic(LISTENING_MSGS[0], LISTENING_CHOICES, self.listening_scores)

    def evaluate_repetitiveness(self):
        control_msg = self.get_control_msg()
        control_msg['text'] = REPETITIVENESS_MSGS[0]
        control_msg['button_choices'] = '</ROUND>'.join(REPETITIVENESS_CHOICES)
        self.eval_agent.observe(validate(control_msg))
        act = self.eval_agent.act(timeout=self.max_resp_time)
        timeout = self.check_timeout(act)
        if timeout:
            return False
        act_choice = REPETITIVENESS_CHOICES.index(act.get('text'))
        self.repetitiveness_scores.append(act_choice)
        if ASK_DETAILED and act_choice != 2:
            control_msg = self.get_control_msg()
            control_msg['text'] = REPETITIVENESS_MSGS[1]
            control_msg['good_rounds'] = True
            control_msg['rounds'] = '</ROUND>'.join(self.dialog_list)
            self.eval_agent.observe(validate(control_msg))
            act = self.eval_agent.act(timeout=self.max_resp_time)
            timeout = self.check_timeout(act)
            if timeout:
                return False
            if 'text' in act:
                self.repetitiveness_scores.append([int(x) - 1 for x in act['text'].split(',')])
        return True

    def evaluate_inquisitiveness(self):
        return self._evaluate_characteristic(INQUISITIVENESS_MSGS[0], INQUISITIVENESS_CHOICES, self.inquisitiveness_scores)

    def evaluate_humanness(self):
        return self._evaluate_characteristic(HUMANNESS_MSGS[0], HUMANNESS_CHOICES, self.humanness_scores)

    def evaluate_fluency(self):
        control_msg = self.get_control_msg()
        control_msg['text'] = FLUENCY_MSGS[0]
        control_msg['button_choices'] = '</ROUND>'.join(FLUENCY_CHOICES)
        self.eval_agent.observe(validate(control_msg))
        act = self.eval_agent.act(timeout=self.max_resp_time)
        timeout = self.check_timeout(act)
        if timeout:
            return False
        act_choice = FLUENCY_CHOICES.index(act.get('text'))
        self.fluency_scores.append(act_choice)
        if ASK_DETAILED and act_choice != 3:
            control_msg = self.get_control_msg()
            control_msg['text'] = FLUENCY_MSGS[1]
            control_msg['good_rounds'] = True
            control_msg['rounds'] = '</ROUND>'.join(self.dialog_list)
            self.eval_agent.observe(validate(control_msg))
            act = self.eval_agent.act(timeout=self.max_resp_time)
            timeout = self.check_timeout(act)
            if timeout:
                return False
            if 'text' in act:
                self.fluency_scores.append([int(x) - 1 for x in act['text'].split(',')])
        return True

    def evaluate_consistency(self):
        control_msg = self.get_control_msg()
        control_msg['text'] = CONSISTENCY_MSGS[0]
        control_msg['button_choices'] = '</ROUND>'.join(CONSISTENCY_CHOICES)
        self.eval_agent.observe(validate(control_msg))
        act = self.eval_agent.act(timeout=self.max_resp_time)
        timeout = self.check_timeout(act)
        if timeout:
            return False
        act_choice = CONSISTENCY_CHOICES.index(act.get('text'))
        self.consistency_scores.append(act_choice)
        if ASK_DETAILED and act_choice != 0:
            control_msg = self.get_control_msg()
            control_msg['text'] = CONSISTENCY_MSGS[1]
            control_msg['good_rounds'] = True
            control_msg['rounds'] = '</ROUND>'.join(self.dialog_list)
            self.eval_agent.observe(validate(control_msg))
            act = self.eval_agent.act(timeout=self.max_resp_time)
            timeout = self.check_timeout(act)
            if timeout:
                return False
            if 'text' in act:
                self.consistency_scores.append([int(x) - 1 for x in act['text'].split(',')])
        return True

    def evaluate_persona(self):
        if self.model_agent is not None:
            other_persona = self.model_personas
        else:
            other_persona = self.other_agent.personas
        fake_persona = self.eval_agent.personas_generator.get_persona()
        while fake_persona == other_persona:
            fake_persona = self.eval_agent.personas_generator.get_persona()
        cand_text = []
        for dt in [other_persona, fake_persona]:
            if dt == other_persona:
                is_correct = True
            else:
                is_correct = False
            _text = ''
            for s in dt:
                _text += '<b><span style="color:blue">' + s.strip() + '</span></b><br>'
            cand_text.append((is_correct, _text))
        random.shuffle(cand_text)
        control_msg = self.get_control_msg()
        control_msg['text'] = PERSONA_MSG.format(cand_text[0][1], cand_text[1][1])
        control_msg['button_choices'] = '</ROUND>'.join(PERSONA_CHOICES)
        self.eval_agent.observe(validate(control_msg))
        act = self.eval_agent.act(timeout=self.max_resp_time)
        timeout = self.check_timeout(act)
        if timeout:
            return False
        self.persona_scores.append(cand_text[int(act['text']) - 1][0])
        return True

    def parallel_eval_mode(self):
        """
        Parallel function that shuts one agent down and asks the other to do the
        evaluation if their are two agents.

        If there is only one agent, it performs the evaluation.
        """

        def eval_or_shutdown(agent):
            if self.model_agent is None and agent == self.other_agent:
                control_msg = self.get_control_msg()
                control_msg['text'] = OTHER_AGENT_FINISHED_MSG
                self.other_agent.observe(validate(control_msg))
                self.eval_agent.mturk_manager.mark_workers_done([self.eval_agent])
                self.other_agent.shutdown()
            else:
                evaluations = [self.evaluate_engagingness, self.evaluate_interestingness, self.evaluate_inquisitiveness, self.evaluate_listening, self.evaluate_repetitiveness, self.evaluate_fluency, self.evaluate_consistency, self.evaluate_humanness, self.evaluate_persona]
                for evaluation in evaluations:
                    fin = evaluation()
                    if not fin:
                        return
                return
        Parallel(n_jobs=len(self.agents), backend='threading')((delayed(eval_or_shutdown)(agent) for agent in self.agents))

    def model_observes_itself(self, txt):
        act = {'text': txt, 'episode_done': False}
        self.model_agent.observe(act)

    def episode_done(self):
        return self.chat_done

    def get_instruction(self, agent_id=None, tag='first'):
        if tag == 'start':
            return START_MSG.format(self.n_turn)
        if tag == 'chat_not_done':
            return CHAT_NOT_DONE_MSG.format(self.n_turn + 1 - self.turn_idx)
        if tag == 'timeout':
            return TIMEOUT_MESSAGE
        if tag == 'exceed_min_turns':
            return EXCEED_MIN_TURNS_MSG.format(self.n_turn)

    def save_data(self):
        convo_finished = True
        bad_workers = []
        if self.dialog == [] or self.persona_scores == []:
            convo_finished = False
        self.convo_finished = convo_finished
        data_path = self.opt['save_data_path']
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        if convo_finished:
            filename = os.path.join(data_path, '{}_{}_{}.json'.format(time.strftime('%Y%m%d-%H%M%S'), np.random.randint(0, 1000), self.task_type))
        else:
            filename = os.path.join(data_path, '{}_{}_{}_incomplete.json'.format(time.strftime('%Y%m%d-%H%M%S'), np.random.randint(0, 1000), self.task_type))
        json.dump({'dialog': self.dialog, 'dialog_list': self.dialog_list, 'other_first': self.other_first, 'bot_went_first': self.other_first, 'start_time': self.start_time, 'timestamp': time.time(), 'total_time': time.time() - self.start_time, 'workers': [ag.worker_id for ag in self.agents], 'hit_id': [ag.hit_id for ag in self.agents], 'assignment_id': [ag.assignment_id for ag in self.agents], 'human_personas': [ag.personas for ag in self.agents], 'model_personas': self.model_personas, 'bad_workers': bad_workers, 'n_turn': self.n_turn, 'engagingness': self.engagingness_scores, 'interestingness': self.interestingness_scores, 'listening': self.listening_scores, 'consistency': self.consistency_scores, 'inquisitiveness': self.inquisitiveness_scores, 'repetitiveness': self.repetitiveness_scores, 'humanness': self.humanness_scores, 'fluency': self.fluency_scores, 'persona': self.persona_scores, 'opt': self.opt, 'model_config': self.model_config}, open(filename, 'w'))
        print(self.world_tag, ': Data successfully saved at {}.'.format(filename))

    def is_msg_tooshortlong(self, act, ag, th_min=3, th_max=20):
        if act['episode_done']:
            return False
        control_msg = self.get_control_msg()
        msg_len = len(act['text'].split(' '))
        if msg_len < th_min:
            control_msg['text'] = TOO_SHORT_MSG.format(th_min)
            ag.observe(validate(control_msg))
            return True
        if msg_len > th_max:
            control_msg['text'] = TOO_LONG_MSG.format(th_max)
            ag.observe(validate(control_msg))
            return True
        return False

    def reset_random(self):
        pass

    def check_timeout(self, act):
        if act is None:
            self.chat_done = True
            return True
        if act['text'] == '[TIMEOUT]' or act['text'] == '[RETURNED]' or act['text'] == '[DISCONNECT]':
            control_msg = self.get_control_msg()
            control_msg['episode_done'] = True
            control_msg['text'] = self.get_instruction(agent_id=act['id'], tag='timeout')
            for ag in self.agents:
                if ag.id != act['id']:
                    ag.observe(validate(control_msg))
            self.chat_done = True
            return True
        else:
            return False

    def shutdown(self):
        self.eval_agent.shutdown()