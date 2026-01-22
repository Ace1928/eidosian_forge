from typing import Dict, List
def add_truecolor_style_rule(is_foreground: bool, ansi_code: int, r: int, g: int, b: int, parameter: str) -> None:
    rule_name = '.ansi{}-{}'.format(ansi_code, parameter)
    color = '#{:02X}{:02X}{:02X}'.format(r, g, b)
    if is_foreground:
        rule = Rule(rule_name, color=color)
    else:
        rule = Rule(rule_name, background_color=color)
    truecolor_rules.append(rule)