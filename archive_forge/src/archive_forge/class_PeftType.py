import enum
class PeftType(str, enum.Enum):
    """
    Enum class for the different types of adapters in PEFT.

    Supported PEFT types:
    - PROMPT_TUNING
    - MULTITASK_PROMPT_TUNING
    - P_TUNING
    - PREFIX_TUNING
    - LORA
    - ADALORA
    - ADAPTION_PROMPT
    - IA3
    - LOHA
    - LOKR
    - OFT
    """
    PROMPT_TUNING = 'PROMPT_TUNING'
    MULTITASK_PROMPT_TUNING = 'MULTITASK_PROMPT_TUNING'
    P_TUNING = 'P_TUNING'
    PREFIX_TUNING = 'PREFIX_TUNING'
    LORA = 'LORA'
    ADALORA = 'ADALORA'
    ADAPTION_PROMPT = 'ADAPTION_PROMPT'
    IA3 = 'IA3'
    LOHA = 'LOHA'
    LOKR = 'LOKR'
    OFT = 'OFT'
    POLY = 'POLY'