import re
def _expand_number(self, m: str) -> str:
    """
        This method acts as a preprocessing step for numbers between 1000 and 3000 (same as the original repository,
        link :
        https://github.com/neonbjb/tortoise-tts/blob/4003544b6ff4b68c09856e04d3eff9da26d023c2/tortoise/utils/tokenizer.py#L86)
        """
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return 'two thousand'
        elif num > 2000 and num < 2010:
            return 'two thousand ' + self.number_to_words(num % 100)
        elif num % 100 == 0:
            return self.number_to_words(num // 100) + ' hundred'
        else:
            return self.number_to_words(num)
    else:
        return self.number_to_words(num)