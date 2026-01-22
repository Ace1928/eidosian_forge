import copy
import json
import os
import random
import re
from collections import defaultdict
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm
from parlai.core.opt import Opt
from parlai.core.teachers import (
from parlai.tasks.convai2.agents import (
from parlai.tasks.empathetic_dialogues.agents import EmpatheticDialoguesTeacher
from parlai.tasks.wizard_of_wikipedia.agents import WizardDialogKnowledgeTeacher
from parlai.utils.misc import warn_once
from .build import build
class PersonaTopicifier:

    def __init__(self, opt: Opt, should_have_personas: bool=False, should_have_topics: bool=False, no_persona_is_error: bool=False):
        self.utterance_to_persona_map = {}
        self.should_have_personas = should_have_personas
        self.should_have_topics = should_have_topics
        self.no_persona_is_error = no_persona_is_error
        self.personas_file_path = _persona_list_path(opt)
        self.topic_to_persona_path = _topic_to_persona_path(opt)
        self.wow_topics_to_persona_strings_map, self.persona_strings_to_wow_topics_map = self._setup_personas_to_wow_topics()
        with open(self.personas_file_path, 'r') as f:
            self.personas = f.read().strip().split('||')
            self.personas = [p for p in self.personas if p]

    def _setup_personas_to_wow_topics(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        persona_strings_to_topics = defaultdict(list)
        topics_to_persona_strings = defaultdict(list)
        with open(self.topic_to_persona_path, 'r') as f:
            for line in f:
                match = re.fullmatch('([^[]+): (\\[.+\\])\\n', line)
                topic = match.group(1)
                persona_strings = eval(match.group(2))
                assert isinstance(persona_strings, list)
                topics_to_persona_strings[topic] = persona_strings
                for str_ in persona_strings:
                    persona_strings_to_topics[str_].append(topic)
        warn_once(f'FINISHED MAPPING personas to topics, got: {len(list(persona_strings_to_topics.keys()))} persona strings to map to topics.')
        return (topics_to_persona_strings, persona_strings_to_topics)

    def __calculate_word_overlap(self, a, b):
        """
        Very rudimentary way to calculate word overlap.
        """
        score = 0
        tokens_a = a.split(' ')
        tokens_a = [ta for ta in tokens_a if len(ta) >= 5]
        for ta in tokens_a:
            if ta in b:
                score += 1
        tokens_b = b.split(' ')
        tokens_b = [tb for tb in tokens_b if len(tb) >= 5]
        for tb in tokens_b:
            if tb in a:
                score += 1
        return score

    def __choose_persona_from_text(self, utt):
        utt = utt.strip()
        if utt not in self.utterance_to_persona_map:
            best_word_overlap = 0
            best_persona = None
            for p in self.personas:
                word_overlap = self.__calculate_word_overlap(utt, p)
                if word_overlap >= best_word_overlap:
                    best_word_overlap = word_overlap
                    best_persona = p
            if not best_persona:
                raise Exception(f'No persona found for utterance: "{utt}". This should not happen.')
            self.utterance_to_persona_map[utt] = best_persona
            return best_persona
        return self.utterance_to_persona_map[utt]

    def __choose_persona_from_topic(self, topic):
        topic = topic.strip()
        persona_strings = self.wow_topics_to_persona_strings_map[topic]
        for p in persona_strings:
            for persona in self.personas:
                if p in persona:
                    return persona
        if self.no_persona_is_error:
            raise ValueError(f'ERROR: Found no persona for topic: {topic}.')
        else:
            warn_once(f'Found no persona for topic: {topic}. Returning first persona.')
            return self.personas[0]

    def __choose_topic(self, persona):
        persona_lines = persona.strip().split('\n')
        for p in persona_lines:
            p_str = p.replace('your persona:', '')
            p_str = p_str.strip()
            if p_str in self.persona_strings_to_wow_topics_map:
                topics = self.persona_strings_to_wow_topics_map[p_str]
                topic = topics[0] + '\n'
                return topic
        for utt, topics in self.persona_strings_to_wow_topics_map.items():
            utt_words = utt.split()
            utt_words_long = [utt for utt in utt_words if len(utt) > 6]
            for long_utt in utt_words_long:
                if long_utt in persona:
                    return topics[0] + '\n'
        return topics[0] + '\n'

    def get_modified_text(self, text):
        has_neither = not self.should_have_personas and (not self.should_have_topics)
        has_wow_topic_only = not self.should_have_personas and self.should_have_topics
        has_persona_only = not self.should_have_topics and self.should_have_personas
        if self.should_have_personas and (has_neither or has_wow_topic_only) or (self.should_have_topics and (has_neither or has_persona_only)):
            raise Exception(f'Malformed text: {text}, should_have_personas: {self.should_have_personas}, should_have_topics: {self.should_have_topics}, has_neither: {has_neither}, has_wow_topic_only: {has_wow_topic_only}, has_persona_only: {has_persona_only}')
        if has_neither:
            persona = self.__choose_persona_from_text(text)
            topic = self.__choose_topic(persona)
            utt = text
        elif has_wow_topic_only:
            parts = text.strip().split('\n')
            if len(parts) > 1:
                topic = parts[0] + '\n'
                utt = parts[1]
                persona = self.__choose_persona_from_topic(topic)
            else:
                topic = parts[0] + '\n'
                utt = ''
                persona = self.__choose_persona_from_topic(topic)
        elif has_persona_only:
            lines = text.strip().split('\n')
            utt = lines[-1]
            persona = ''.join((l + '\n' for l in lines[:-1]))
            topic = self.__choose_topic(persona)
        else:
            raise Exception(f'Unknown structure of utterance: {text}')
        modified_utterance = persona + topic + utt
        return modified_utterance