import sys
import time
from parlai.core.script import ParlaiScript, register_script
from parlai.core.params import ParlaiParser
@register_script('party', hidden=True, aliases=['parrot'])
class Party(ParlaiScript):

    @classmethod
    def setup_args(cls):
        return ParlaiParser(False, False, 'Throw a party!')

    def run(self):
        i = 0
        while True:
            try:
                frame = FRAMES[i % len(FRAMES)]
                color = COLORS[i % len(COLORS)]
                i += 1
                sys.stdout.write(CLEAR_SCREEN)
                sys.stdout.write(color)
                sys.stdout.write('\n\n    ')
                sys.stdout.write(frame.replace('\n', '\n    '))
                sys.stdout.write(COLORS[(i * 3 + 1) % len(COLORS)])
                sys.stdout.write('\n\n              P A R T Y    P A R R O T\n')
                sys.stdout.write(RESET)
                time.sleep(75 / 1000)
            except KeyboardInterrupt:
                sys.stdout.write(RESET + '\n')
                break