from __future__ import annotations
import json
from typing import Dict, Literal, TypedDict
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from typing_extensions import NotRequired
from speech_recognition.audio import AudioData
from speech_recognition.exceptions import RequestError, UnknownValueError
class OutputParser:

    def __init__(self, *, show_all: bool, with_confidence: bool) -> None:
        self.show_all = show_all
        self.with_confidence = with_confidence

    def parse(self, response_text: str):
        actual_result = self.convert_to_result(response_text)
        if self.show_all:
            return actual_result
        best_hypothesis = self.find_best_hypothesis(actual_result['alternative'])
        confidence = best_hypothesis.get('confidence', 0.5)
        if self.with_confidence:
            return (best_hypothesis['transcript'], confidence)
        return best_hypothesis['transcript']

    @staticmethod
    def convert_to_result(response_text: str) -> Result:
        """
        >>> response_text = '''{"result":[]}
        ... {"result":[{"alternative":[{"transcript":"one two three","confidence":0.49585345},{"transcript":"1 2","confidence":0.42899391}],"final":true}],"result_index":0}
        ... '''
        >>> OutputParser.convert_to_result(response_text)
        {'alternative': [{'transcript': 'one two three', 'confidence': 0.49585345}, {'transcript': '1 2', 'confidence': 0.42899391}], 'final': True}

        >>> OutputParser.convert_to_result("")
        Traceback (most recent call last):
          ...
        speech_recognition.exceptions.UnknownValueError
        >>> OutputParser.convert_to_result('\\n{"result":[]}')
        Traceback (most recent call last):
          ...
        speech_recognition.exceptions.UnknownValueError
        >>> OutputParser.convert_to_result('{"result":[{"foo": "bar"}]}')
        Traceback (most recent call last):
          ...
        speech_recognition.exceptions.UnknownValueError
        >>> OutputParser.convert_to_result('{"result":[{"alternative": []}]}')
        Traceback (most recent call last):
          ...
        speech_recognition.exceptions.UnknownValueError
        """
        for line in response_text.split('\n'):
            if not line:
                continue
            result: list[Result] = json.loads(line)['result']
            if len(result) != 0:
                if len(result[0].get('alternative', [])) == 0:
                    raise UnknownValueError()
                return result[0]
        raise UnknownValueError()

    @staticmethod
    def find_best_hypothesis(alternatives: list[Alternative]) -> Alternative:
        """
        >>> alternatives = [{"transcript": "one two three", "confidence": 0.42899391}, {"transcript": "1 2", "confidence": 0.49585345}]
        >>> OutputParser.find_best_hypothesis(alternatives)
        {'transcript': 'one two three', 'confidence': 0.42899391}

        >>> alternatives = [{"confidence": 0.49585345}]
        >>> OutputParser.find_best_hypothesis(alternatives)
        Traceback (most recent call last):
          ...
        speech_recognition.exceptions.UnknownValueError
        """
        if 'confidence' in alternatives:
            best_hypothesis: Alternative = max(alternatives, key=lambda alternative: alternative['confidence'])
        else:
            best_hypothesis: Alternative = alternatives[0]
        if 'transcript' not in best_hypothesis:
            raise UnknownValueError()
        return best_hypothesis