from typing import Optional, Callable, Iterable, List, Dict
from pydantic import BaseModel
def tokenize_conversations(self, conversations: Iterable[Conversation], inference: bool=False, seq_level_weight: bool=False):
    default_condition = self.inference_condition if inference else ''
    sys_mappings = set()
    role_mappings = set()
    all_text = []
    for conv in conversations:
        sys_mappings.add(conv.system)
        for msg in conv.items:
            role_mappings.add((msg.role, conv.condition or default_condition))
            all_text.append(msg.content)
    sys_mappings = list(sys_mappings)
    role_mappings = list(role_mappings)
    sys_mappings = dict(zip(sys_mappings, self._tokenize(sys_mappings)))
    role_mappings = dict(zip(role_mappings, self._tokenize([self.role_prefix(*args) for args in role_mappings], ignore_special=False)))
    all_text = self._tokenize(all_text)
    result_tokens = []
    result_weights = []
    all_text_idx = 0
    for conv in conversations:
        tokens = []
        weights = []
        tokens.extend(self.bos_tokens_)
        weights.extend([0.0] * len(self.bos_tokens_))
        if conv.system:
            system = sys_mappings[conv.system]
            tokens.extend(system)
            weights.extend([0.0] * len(system))
            tokens.extend(self.eot_tokens_)
            weights.extend([0.0] * len(self.eot_tokens_))
        last_idx = len(conv.items) - 1
        for idx, msg in enumerate(conv.items):
            role = role_mappings[msg.role, conv.condition or default_condition]
            tokens.extend(role)
            weights.extend([0.0] * len(role))
            text = all_text[all_text_idx]
            all_text_idx += 1
            w = None
            if not inference:
                assert msg.weight is not None
                w = msg.weight
                if seq_level_weight:
                    w /= len(text) + len(self.eot_tokens_)
            tokens.extend(text)
            weights.extend([w] * len(text))
            if not (inference and idx == last_idx):
                tokens.extend(self.eot_tokens_)
                weights.extend([w] * len(self.eot_tokens_))
        result_tokens.append(tokens)
        result_weights.append(weights)
    assert all_text_idx == len(all_text)
    return (result_tokens, result_weights)