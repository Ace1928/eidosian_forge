import sys
import threading
import wandb
def _windows_timed_input(prompt: str, timeout: float) -> str:
    interval = 0.1
    _echo(prompt)
    begin = time.monotonic()
    end = begin + timeout
    line = ''
    while time.monotonic() < end:
        if msvcrt.kbhit():
            c = msvcrt.getwche()
            if c in (CR, LF):
                _echo(CRLF)
                return line
            if c == '\x03':
                raise KeyboardInterrupt
            if c == '\x08':
                line = line[:-1]
                cover = SP * len(prompt + line + SP)
                _echo(''.join([CR, cover, CR, prompt, line]))
            else:
                line += c
        time.sleep(interval)
    _echo(CRLF)
    raise TimeoutError