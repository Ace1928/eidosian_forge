import math
import copy
import random
import sys
import threading
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict
import string

# ------------------------------
# Core Structures
# ------------------------------


class Codon:
    VALID_INSTRUCTIONS = [
        'LOAD', 'ADD', 'SUB', 'MULT', 'DIV', 'MOD',
        'JMP', 'IF', 'PRINT', 'NOP', 'AND', 'OR',
        'NOT', 'XOR', 'MOVE', 'COPY', 'EXP', 'SQRT',
        'LOG', 'CREATE_ARRAY', 'SET_INDEX', 'GET_INDEX',
        'DEFINE_FUNCTION', 'CALL_FUNCTION', 'RETURN',
        'TRY', 'CATCH', 'THROW', 'PUSH', 'POP'
    ]

    def __init__(self, instruction: str, data: List[Any], modifier: str = "default"):
        self.instruction = instruction.upper()  # e.g., "ADD", "LOAD"
        self.data = data                        # Operands or arguments
        self.modifier = modifier                # Epigenetic effect
        self.previous_instruction = self.instruction  # For mutation rollback

    def mutate_instruction(self, mutation_rate: float = 0.05):
        """Mutate the instruction by replacing it with another valid instruction based on mutation rate."""
        if random.random() < mutation_rate:
            previous_instruction = self.instruction
            possible_mutations = [instr for instr in self.VALID_INSTRUCTIONS if instr != self.instruction]
            if possible_mutations:
                self.instruction = random.choice(possible_mutations)
                print(f"Mutation: Instruction changed from '{previous_instruction}' to '{self.instruction}'.")

    def mutate_modifier(self, mutation_rate: float = 0.1):
        """Randomly change the modifier based on mutation rate."""
        if random.random() < mutation_rate:
            previous_modifier = self.modifier
            modifiers = ["default", "conditional", "loop", "activate_if", "suppress_if"]
            self.modifier = random.choice(modifiers)
            print(f"Mutation: Modifier changed from '{previous_modifier}' to '{self.modifier}'.")

    def mutate_data(self, mutation_rate: float = 0.2):
        """Randomly mutate one of the data elements based on mutation rate."""
        if not self.data:
            return  # No data to mutate

        if random.random() < mutation_rate:
            index = random.randint(0, len(self.data) - 1)
            original_value = self.data[index]
            if isinstance(original_value, int):
                # Mutate integer by adding a small random value
                mutation = random.randint(-5, 5)
                self.data[index] = max(0, original_value + mutation)
                print(f"Mutation: Data at index {index} changed from {original_value} to {self.data[index]}.")
            elif isinstance(original_value, str):
                # Mutate string by randomly changing a character
                if original_value:
                    char_list = list(original_value)
                    char_index = random.randint(0, len(char_list) - 1)
                    original_char = char_list[char_index]
                    char_list[char_index] = random.choice(string.ascii_lowercase)
                    mutated_str = ''.join(char_list)
                    self.data[index] = mutated_str
                    print(f"Mutation: Data at index {index} changed from '{original_char}' to '{char_list[char_index]}' in '{original_value}'.")

    def mutate(self, mutation_rates: Dict[str, float] = None):
        """Apply mutations to the codon with specified mutation rates."""
        if mutation_rates is None:
            mutation_rates = {
                'instruction': 0.05,  # 5% mutation rate
                'modifier': 0.1,      # 10% mutation rate
                'data': 0.2           # 20% mutation rate
            }

        self.mutate_instruction(mutation_rate=mutation_rates.get('instruction', 0.05))
        self.mutate_modifier(mutation_rate=mutation_rates.get('modifier', 0.1))
        self.mutate_data(mutation_rate=mutation_rates.get('data', 0.2))

    def to_dict(self) -> Dict[str, Any]:
        """Convert Codon instance to dictionary."""
        return {
            'instruction': self.instruction,
            'data': self.data,
            'modifier': self.modifier
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Codon':
        """Create Codon instance from dictionary."""
        return Codon(
            instruction=data['instruction'],
            data=data['data'],
            modifier=data.get('modifier', 'default')
        )


class Gene:
    def __init__(self, promoter: str, codons: List[Codon], terminator: str, modifier: Optional[str] = None):
        self.promoter = promoter.lower()        # e.g., "start_reflex"
        self.codons = codons                    # List of Codon instances
        self.terminator = terminator.lower()    # e.g., "end_reflex"
        self.active = False                      # Gene activation status
        self.modifier = modifier
        self.previous_promoter = self.promoter  # For mutation rollback
        self.previous_terminator = self.terminator  # For mutation rollback

    def mutate_gene(self, mutation_rates: Dict[str, float] = None):
        """Mutate the gene by mutating its codons and potentially its promoter or terminator."""
        if mutation_rates is None:
            mutation_rates = {
                'codon': 0.2,          # 20% chance to mutate each codon
                'promoter': 0.02,     # 2% chance to mutate promoter
                'terminator': 0.02     # 2% chance to mutate terminator
            }

        # Mutate codons
        for codon in self.codons:
            codon.mutate(mutation_rates={
                'instruction': mutation_rates.get('instruction', 0.05),
                'modifier': mutation_rates.get('modifier', 0.1),
                'data': mutation_rates.get('data', 0.2)
            })

        # Possibly mutate promoter
        if random.random() < mutation_rates.get('promoter', 0.02):
            new_promoter = ''.join(random.choices(string.ascii_lowercase, k=10))
            print(f"Mutation: Promoter changed from '{self.promoter}' to '{new_promoter}'.")
            self.previous_promoter = self.promoter
            self.promoter = new_promoter.lower()

        # Possibly mutate terminator
        if random.random() < mutation_rates.get('terminator', 0.02):
            new_terminator = ''.join(random.choices(string.ascii_lowercase, k=10))
            print(f"Mutation: Terminator changed from '{self.terminator}' to '{new_terminator}'.")
            self.previous_terminator = self.terminator
            self.terminator = new_terminator.lower()

    def to_dict(self) -> Dict[str, Any]:
        """Convert Gene instance to dictionary."""
        return {
            'promoter': self.promoter,
            'codons': [codon.to_dict() for codon in self.codons],
            'terminator': self.terminator,
            'active': self.active,
            'modifier': self.modifier
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Gene':
        """Create Gene instance from dictionary."""
        codons = [Codon.from_dict(codon_data) for codon_data in data['codons']]
        gene = Gene(
            promoter=data['promoter'],
            codons=codons,
            terminator=data['terminator'],
            modifier=data.get('modifier')
        )
        gene.active = data.get('active', False)
        return gene


class Genome:
    def __init__(self, functional_genes: List[Gene], control_genes: List[Gene], interpreter_genes: List[Gene]):
        self.functional_genes = functional_genes
        self.control_genes = control_genes
        self.interpreter_genes = interpreter_genes

    def mutate_genome(self, mutation_rates: Dict[str, float] = None):
        """Apply mutations to the entire genome."""
        if mutation_rates is None:
            mutation_rates = {
                'functional_gene': {
                    'codon': 0.2,
                    'promoter': 0.02,
                    'terminator': 0.02
                },
                'control_gene': {
                    'codon': 0.2,
                    'promoter': 0.02,
                    'terminator': 0.02
                },
                'interpreter_gene': {
                    'codon': 0.2,
                    'promoter': 0.02,
                    'terminator': 0.02
                },
                'genome_level': {
                    'add_gene': 0.005,
                    'remove_gene': 0.005
                }
            }

        # Mutate functional genes
        for gene in self.functional_genes:
            gene.mutate_gene(mutation_rates=mutation_rates.get('functional_gene', {}))

        # Mutate control genes
        for gene in self.control_genes:
            gene.mutate_gene(mutation_rates=mutation_rates.get('control_gene', {}))

        # Mutate interpreter genes
        for gene in self.interpreter_genes:
            gene.mutate_gene(mutation_rates=mutation_rates.get('interpreter_gene', {}))

        # Genome-level mutations
        # Add a new functional gene
        if random.random() < mutation_rates.get('genome_level', {}).get('add_gene', 0.005):
            new_gene = Gene(
                promoter="start_new_gene",
                codons=[
                    Codon("LOAD", ["data", "y", 10], "default"),
                    Codon("ADD", ["y", "x", "y"], "conditional"),  # Corrected data length from 4 to 3
                    Codon("PRINT", ["y"], "default")
                ],
                terminator="end_new_gene"
            )
            self.functional_genes.append(new_gene)
            print("Mutation: Added a new functional gene.")

        # Remove a functional gene
        if random.random() < mutation_rates.get('genome_level', {}).get('remove_gene', 0.005) and self.functional_genes:
            removed_gene = self.functional_genes.pop(random.randint(0, len(self.functional_genes)-1))
            print(f"Mutation: Removed a functional gene with promoter '{removed_gene.promoter}'.")

    def crossover(self, other_genome: 'Genome', crossover_rate: float = 0.3) -> 'Genome':
        """Perform crossover with another genome to produce a new genome."""
        child_functional = copy.deepcopy(self.functional_genes)
        child_control = copy.deepcopy(self.control_genes)
        child_interpreter = copy.deepcopy(self.interpreter_genes)

        # Crossover functional genes
        for gene in other_genome.functional_genes:
            if random.random() < crossover_rate:
                child_functional.append(copy.deepcopy(gene))

        # Crossover control genes
        for gene in other_genome.control_genes:
            if random.random() < crossover_rate:
                child_control.append(copy.deepcopy(gene))

        # Crossover interpreter genes
        for gene in other_genome.interpreter_genes:
            if random.random() < crossover_rate:
                child_interpreter.append(copy.deepcopy(gene))

        return Genome(child_functional, child_control, child_interpreter)

    def duplicate_gene(self, gene_type: str):
        """Duplicate a gene within a specified gene type."""
        gene_list = getattr(self, gene_type, None)
        if gene_list and gene_list:
            gene_to_duplicate = random.choice(gene_list)
            duplicated_gene = copy.deepcopy(gene_to_duplicate)
            # Modify promoter and terminator to maintain uniqueness
            duplicated_gene.promoter = ''.join(random.choices(string.ascii_lowercase, k=10))
            duplicated_gene.terminator = f"end_{duplicated_gene.promoter}"
            gene_list.append(duplicated_gene)
            print(f"Duplication: Duplicated a '{gene_type}' gene with new promoter '{duplicated_gene.promoter}'.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert Genome instance to dictionary."""
        return {
            'functional_genes': [gene.to_dict() for gene in self.functional_genes],
            'control_genes': [gene.to_dict() for gene in self.control_genes],
            'interpreter_genes': [gene.to_dict() for gene in self.interpreter_genes]
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Genome':
        """Create Genome instance from dictionary."""
        functional_genes = [Gene.from_dict(gene_data) for gene_data in data['functional_genes']]
        control_genes = [Gene.from_dict(gene_data) for gene_data in data['control_genes']]
        interpreter_genes = [Gene.from_dict(gene_data) for gene_data in data['interpreter_genes']]
        return Genome(functional_genes, control_genes, interpreter_genes)


# ------------------------------
# Genetic Interpreter
# ------------------------------

class GeneticInterpreter:
    VALID_INSTRUCTIONS = Codon.VALID_INSTRUCTIONS

    def __init__(self, genome: Genome):
        self.genome = genome
        self.memory = defaultdict(dict)
        self.memory['stack'] = []
        self.memory['heap'] = {}
        self.memory['data'] = {}
        self.memory['call_stack'] = []
        self.memory['functions'] = {}
        self.memory['exception_stack'] = []
        self.memory['exception_handlers'] = []
        self.memory['exceptions'] = []
        self.execution_log = []
        self.running = True  # To control the interpreter's execution
        self.current_gene_promoter = None  # Track the current gene for call stack

        # Initialize symbol mappings
        self.initialize_symbol_mappings()

        # Initialize mutation rates for adaptive mutation
        self.mutation_rates = {
            'codon': 0.2,          # 20% chance to mutate each codon
            'promoter': 0.02,     # 2% chance to mutate promoter
            'terminator': 0.02     # 2% chance to mutate terminator
        }

    def initialize_symbol_mappings(self):
        """Initialize all symbol mappings."""
        # English Alphabet Mapping
        self.ENGLISH_ALPHABET = {chr(i + 96): i for i in range(1, 27)}  # a=1, b=2, ..., z=26

        # Phonics and Phoneme Mapping
        self.PHONEMES = {
            'a': 'æ',
            'b': 'b',
            'c': 'k',
            'd': 'd',
            'e': 'ɛ',
            'f': 'f',
            'g': 'g',
            'h': 'h',
            'i': 'ɪ',
            'j': 'dʒ',
            'k': 'k',
            'l': 'l',
            'm': 'm',
            'n': 'n',
            'o': 'ɑ',
            'p': 'p',
            'q': 'kw',
            'r': 'r',
            's': 's',
            't': 't',
            'u': 'ʌ',
            'v': 'v',
            'w': 'w',
            'x': 'ks',
            'y': 'j',
            'z': 'z'
        }

        # Logic and Mathematics Symbols Mapping
        self.LOGIC_SYMBOLS = {
            '&&': 'AND',
            '||': 'OR',
            '!': 'NOT',
            '^': 'XOR'
        }

        self.MATH_SYMBOLS = {
            '+': 'ADD',
            '-': 'SUB',
            '*': 'MULT',
            '/': 'DIV',
            '%': 'MOD',
            '^': 'EXP',
            'sqrt': 'SQRT',
            'log': 'LOG',
            'exp': 'EXP'
        }

    # ------------------------------
    # Symbol Mapping Functions
    # ------------------------------
    def encode_letter(self, letter: str) -> int:
        """Encode an English letter to its numerical equivalent."""
        return self.ENGLISH_ALPHABET.get(letter.lower(), 0)

    def decode_number(self, number: int) -> str:
        """Decode a number back to its corresponding English letter."""
        for letter, num in self.ENGLISH_ALPHABET.items():
            if num == number:
                return letter
        return ''

    def encode_phoneme(self, letter: str) -> str:
        """Encode an English letter to its phoneme."""
        return self.PHONEMES.get(letter.lower(), '')

    def decode_phoneme(self, phoneme: str) -> str:
        """Decode a phoneme back to its corresponding English letter."""
        for letter, ph in self.PHONEMES.items():
            if ph == phoneme:
                return letter
        return ''

    def symbol_to_instruction(self, symbol: str) -> Optional[str]:
        """Convert a symbol to its corresponding instruction name."""
        if symbol in self.LOGIC_SYMBOLS:
            return self.LOGIC_SYMBOLS[symbol]
        if symbol in self.MATH_SYMBOLS:
            return self.MATH_SYMBOLS[symbol]
        return None

    def instruction_to_symbol(self, instruction: str) -> Optional[str]:
        """Convert an instruction name back to its corresponding symbol."""
        for sym, instr in self.LOGIC_SYMBOLS.items():
            if instr == instruction:
                return sym
        for sym, instr in self.MATH_SYMBOLS.items():
            if instr == instruction:
                return sym
        return None

    # ------------------------------
    # Instruction Implementations
    # ------------------------------

    def execute_codon(self, codon: Codon):
        """Execute a single codon."""
        if not self.validate_instruction(codon):
            self.execution_log.append(f"Invalid instruction '{codon.instruction}'. Skipping codon.")
            print(f"Invalid instruction '{codon.instruction}'. Skipping codon.")
            return
        try:
            instruction = codon.instruction.upper()
            method_name = f"instr_{instruction}"
            method = getattr(self, method_name, self.instr_NOP)
            method(codon.data, codon.modifier)
        except Exception as e:
            self.execution_log.append(f"Error executing instruction '{instruction}': {e}")
            print(f"Error executing instruction '{instruction}': {e}")
            self.running = False  # Halt execution due to error

    def validate_instruction(self, codon: Codon) -> bool:
        """Validate instruction and data before execution."""
        if codon.instruction not in self.VALID_INSTRUCTIONS:
            return False
        # Additional data validation can be implemented here
        return True

    def instr_LOAD(self, data: List[Any], modifier: str):
        """Load data into a specific memory segment."""
        if len(data) != 3:
            self.execution_log.append("LOAD: Invalid data length.")
            print("LOAD: Invalid data length.")
            return
        segment, key, value = data
        if segment not in self.memory:
            self.execution_log.append(f"LOAD: Invalid memory segment '{segment}'.")
            print(f"LOAD: Invalid memory segment '{segment}'.")
            return
        if isinstance(value, int) or isinstance(value, bool):
            self.memory[segment][key] = value
            self.execution_log.append(f"LOAD: {key} = {value} in '{segment}'.")
            print(f"LOAD: {key} = {value} in '{segment}'.")
        else:
            self.execution_log.append("LOAD: Value must be integer or boolean.")
            print("LOAD: Value must be integer or boolean.")

    def instr_ADD(self, data: List[Any], modifier: str):
        """Add two numbers and store the result."""
        if len(data) != 3:
            self.execution_log.append("ADD: Invalid data length.")
            print("ADD: Invalid data length.")
            return
        dest, op1, op2 = data
        val1 = self.memory['data'].get(op1, 0)
        val2 = self.memory['data'].get(op2, 0)
        self.memory['data'][dest] = val1 + val2
        self.execution_log.append(f"ADD: {dest} = {op1}({val1}) + {op2}({val2}) = {self.memory['data'][dest]}")
        print(f"ADD: {dest} = {op1}({val1}) + {op2}({val2}) = {self.memory['data'][dest]}")

    def instr_SUB(self, data: List[Any], modifier: str):
        """Subtract two numbers and store the result."""
        if len(data) != 3:
            self.execution_log.append("SUB: Invalid data length.")
            print("SUB: Invalid data length.")
            return
        dest, op1, op2 = data
        val1 = self.memory['data'].get(op1, 0)
        val2 = self.memory['data'].get(op2, 0)
        self.memory['data'][dest] = val1 - val2
        self.execution_log.append(f"SUB: {dest} = {op1}({val1}) - {op2}({val2}) = {self.memory['data'][dest]}")
        print(f"SUB: {dest} = {op1}({val1}) - {op2}({val2}) = {self.memory['data'][dest]}")

    def instr_MULT(self, data: List[Any], modifier: str):
        """Multiply two numbers and store the result."""
        if len(data) != 3:
            self.execution_log.append("MULT: Invalid data length.")
            print("MULT: Invalid data length.")
            return
        dest, op1, op2 = data
        val1 = self.memory['data'].get(op1, 0)
        val2 = self.memory['data'].get(op2, 0)
        self.memory['data'][dest] = val1 * val2
        self.execution_log.append(f"MULT: {dest} = {op1}({val1}) * {op2}({val2}) = {self.memory['data'][dest]}")
        print(f"MULT: {dest} = {op1}({val1}) * {op2}({val2}) = {self.memory['data'][dest]}")

    def instr_DIV(self, data: List[Any], modifier: str):
        """Divide two numbers and store the result."""
        if len(data) != 3:
            self.execution_log.append("DIV: Invalid data length.")
            print("DIV: Invalid data length.")
            return
        dest, op1, op2 = data
        val1 = self.memory['data'].get(op1, 0)
        val2 = self.memory['data'].get(op2, 1)  # Prevent division by zero
        if val2 == 0:
            self.execution_log.append("DIV: Division by zero.")
            print("DIV: Division by zero.")
            return
        self.memory['data'][dest] = val1 / val2
        self.execution_log.append(f"DIV: {dest} = {op1}({val1}) / {op2}({val2}) = {self.memory['data'][dest]}")
        print(f"DIV: {dest} = {op1}({val1}) / {op2}({val2}) = {self.memory['data'][dest]}")

    def instr_MOD(self, data: List[Any], modifier: str):
        """Modulo operation and store the result."""
        if len(data) != 3:
            self.execution_log.append("MOD: Invalid data length.")
            print("MOD: Invalid data length.")
            return
        dest, op1, op2 = data
        val1 = self.memory['data'].get(op1, 0)
        val2 = self.memory['data'].get(op2, 1)  # Prevent modulo by zero
        if val2 == 0:
            self.execution_log.append("MOD: Modulo by zero.")
            print("MOD: Modulo by zero.")
            return
        self.memory['data'][dest] = val1 % val2
        self.execution_log.append(f"MOD: {dest} = {op1}({val1}) % {op2}({val2}) = {self.memory['data'][dest]}")
        print(f"MOD: {dest} = {op1}({val1}) % {op2}({val2}) = {self.memory['data'][dest]}")

    def instr_JMP(self, data: List[Any], modifier: str):
        """Jump to a specific gene based on modifier conditions."""
        if len(data) != 1:
            self.execution_log.append("JMP: Invalid data length.")
            print("JMP: Invalid data length.")
            return
        target_promoter = data[0].lower()
        promoters = [gene.promoter for gene in self.genome.functional_genes]
        if target_promoter in promoters:
            # Activate the target gene and deactivate others
            for gene in self.genome.functional_genes:
                gene.active = (gene.promoter == target_promoter)
            self.execution_log.append(f"JMP: Jumping to gene with promoter '{target_promoter}'.")
            print(f"JMP: Jumping to gene with promoter '{target_promoter}'.")
        else:
            self.execution_log.append(f"JMP: Target promoter '{target_promoter}' not found.")
            print(f"JMP: Target promoter '{target_promoter}' not found.")

    def instr_IF(self, data: List[Any], modifier: str):
        """Conditional execution based on a condition."""
        if len(data) < 2:
            self.execution_log.append("IF: Invalid data length.")
            print("IF: Invalid data length.")
            return
        condition = data[0].lower()
        target_promoter = data[1].lower()
        condition_met = False

        # Simple condition parsing
        if condition.startswith("x>"):
            try:
                threshold = float(condition[2:])
                condition_met = self.memory['data'].get("x", 0) > threshold
            except ValueError:
                pass
        elif condition.startswith("y<"):
            try:
                threshold = float(condition[2:])
                condition_met = self.memory['data'].get("y", 0) < threshold
            except ValueError:
                pass
        elif condition.startswith("z=="):
            try:
                value = float(condition[3:])
                condition_met = self.memory['data'].get("z", 0) == value
            except ValueError:
                pass
        # Add more conditions as needed

        if condition_met:
            self.instr_JMP([target_promoter], modifier)

    def instr_PRINT(self, data: List[Any], modifier: str):
        """Print a value from memory."""
        if len(data) != 1:
            self.execution_log.append("PRINT: Invalid data length.")
            print("PRINT: Invalid data length.")
            return
        key = data[0]
        value = self.memory['data'].get(key, None)
        self.execution_log.append(f"PRINT: {key} = {value}")
        print(f"PRINT: {key} = {value}")

    def instr_NOP(self, data: List[Any], modifier: str):
        """No Operation."""
        self.execution_log.append("NOP: No operation performed.")
        print("NOP: No operation performed.")

    # ------------------------------
    # Phase 1: Foundation Building
    # ------------------------------

    # Stack Operations
    def instr_PUSH(self, data: List[Any], modifier: str):
        """Push a value onto the stack."""
        if len(data) != 1:
            self.execution_log.append("PUSH: Invalid data length.")
            print("PUSH: Invalid data length.")
            return
        value = data[0]
        if isinstance(value, int):
            self.memory['stack'].append(value)
            self.execution_log.append(f"PUSH: Pushed {value} onto the stack.")
            print(f"PUSH: Pushed {value} onto the stack.")
        else:
            self.execution_log.append("PUSH: Value must be integer.")
            print("PUSH: Value must be integer.")

    def instr_POP(self, data: List[Any], modifier: str):
        """Pop a value from the stack into a specified memory location."""
        if len(data) != 1:
            self.execution_log.append("POP: Invalid data length.")
            print("POP: Invalid data length.")
            return
        key = data[0]
        if self.memory['stack']:
            value = self.memory['stack'].pop()
            self.memory['data'][key] = value
            self.execution_log.append(f"POP: Popped {value} from the stack into {key}.")
            print(f"POP: Popped {value} from the stack into {key}.")
        else:
            self.execution_log.append("POP: Stack is empty.")
            print("POP: Stack is empty.")

    # Logical Operations
    def instr_AND(self, data: List[Any], modifier: str):
        """Logical AND operation."""
        if len(data) != 3:
            self.execution_log.append("AND: Invalid data length.")
            print("AND: Invalid data length.")
            return
        dest, op1, op2 = data
        val1 = bool(self.memory['data'].get(op1, False))
        val2 = bool(self.memory['data'].get(op2, False))
        self.memory['data'][dest] = val1 and val2
        self.execution_log.append(f"AND: {dest} = {op1}({val1}) AND {op2}({val2}) = {self.memory['data'][dest]}")
        print(f"AND: {dest} = {op1}({val1}) AND {op2}({val2}) = {self.memory['data'][dest]}")

    def instr_OR(self, data: List[Any], modifier: str):
        """Logical OR operation."""
        if len(data) != 3:
            self.execution_log.append("OR: Invalid data length.")
            print("OR: Invalid data length.")
            return
        dest, op1, op2 = data
        val1 = bool(self.memory['data'].get(op1, False))
        val2 = bool(self.memory['data'].get(op2, False))
        self.memory['data'][dest] = val1 or val2
        self.execution_log.append(f"OR: {dest} = {op1}({val1}) OR {op2}({val2}) = {self.memory['data'][dest]}")
        print(f"OR: {dest} = {op1}({val1}) OR {op2}({val2}) = {self.memory['data'][dest]}")

    def instr_NOT(self, data: List[Any], modifier: str):
        """Logical NOT operation."""
        if len(data) != 2:
            self.execution_log.append("NOT: Invalid data length.")
            print("NOT: Invalid data length.")
            return
        dest, op = data
        val = bool(self.memory['data'].get(op, False))
        self.memory['data'][dest] = not val
        self.execution_log.append(f"NOT: {dest} = NOT {op}({val}) = {self.memory['data'][dest]}")
        print(f"NOT: {dest} = NOT {op}({val}) = {self.memory['data'][dest]}")

    def instr_XOR(self, data: List[Any], modifier: str):
        """Logical XOR operation."""
        if len(data) != 3:
            self.execution_log.append("XOR: Invalid data length.")
            print("XOR: Invalid data length.")
            return
        dest, op1, op2 = data
        val1 = bool(self.memory['data'].get(op1, False))
        val2 = bool(self.memory['data'].get(op2, False))
        self.memory['data'][dest] = val1 != val2
        self.execution_log.append(f"XOR: {dest} = {op1}({val1}) XOR {op2}({val2}) = {self.memory['data'][dest]}")
        print(f"XOR: {dest} = {op1}({val1}) XOR {op2}({val2}) = {self.memory['data'][dest]}")

    # Advanced Mathematical Operations
    def instr_EXP(self, data: List[Any], modifier: str):
        """Exponentiation operation."""
        if len(data) != 3:
            self.execution_log.append("EXP: Invalid data length.")
            print("EXP: Invalid data length.")
            return
        dest, base, exponent = data
        val1 = self.memory['data'].get(base, 0)
        val2 = self.memory['data'].get(exponent, 0)
        try:
            result = math.pow(val1, val2)
            self.memory['data'][dest] = result
            self.execution_log.append(f"EXP: {dest} = {base}({val1}) ^ {exponent}({val2}) = {result}")
            print(f"EXP: {dest} = {base}({val1}) ^ {exponent}({val2}) = {result}")
        except OverflowError:
            self.execution_log.append("EXP: Overflow error during exponentiation.")
            print("EXP: Overflow error during exponentiation.")

    def instr_SQRT(self, data: List[Any], modifier: str):
        """Square root operation."""
        if len(data) != 2:
            self.execution_log.append("SQRT: Invalid data length.")
            print("SQRT: Invalid data length.")
            return
        dest, src = data
        src_val = self.memory['data'].get(src, 0)
        if src_val < 0:
            self.execution_log.append("SQRT: Negative value encountered.")
            print("SQRT: Negative value encountered.")
            return
        result = math.sqrt(src_val)
        self.memory['data'][dest] = result
        self.execution_log.append(f"SQRT: {dest} = sqrt({src}({src_val})) = {result}")
        print(f"SQRT: {dest} = sqrt({src}({src_val})) = {result}")

    def instr_LOG(self, data: List[Any], modifier: str):
        """Natural logarithm operation."""
        if len(data) != 2:
            self.execution_log.append("LOG: Invalid data length.")
            print("LOG: Invalid data length.")
            return
        dest, src = data
        src_val = self.memory['data'].get(src, 0)
        if src_val <= 0:
            self.execution_log.append("LOG: Non-positive value encountered.")
            print("LOG: Non-positive value encountered.")
            return
        result = math.log(src_val)
        self.memory['data'][dest] = result
        self.execution_log.append(f"LOG: {dest} = log({src}({src_val})) = {result}")
        print(f"LOG: {dest} = log({src}({src_val})) = {result}")

    # Data Movement Operations
    def instr_MOVE(self, data: List[Any], modifier: str):
        """Move data from one memory location to another."""
        if len(data) != 2:
            self.execution_log.append("MOVE: Invalid data length.")
            print("MOVE: Invalid data length.")
            return
        dest, src = data
        src_value = self.get_memory_value(src)
        if src_value is None:
            self.execution_log.append(f"MOVE: Source '{src}' not found.")
            print(f"MOVE: Source '{src}' not found.")
            return
        # Minimal Change: Set in 'heap' if 'heap' is in dest, else in 'data'
        if 'heap' in dest:
            self.memory['heap'][dest] = src_value
        else:
            self.memory['data'][dest] = src_value
        self.execution_log.append(f"MOVE: Moved {src}({src_value}) to {dest}.")
        print(f"MOVE: Moved {src}({src_value}) to {dest}.")

    def instr_COPY(self, data: List[Any], modifier: str):
        """Copy data from one memory location to another."""
        if len(data) != 2:
            self.execution_log.append("COPY: Invalid data length.")
            print("COPY: Invalid data length.")
            return
        dest, src = data
        src_value = self.get_memory_value(src)
        if src_value is None:
            self.execution_log.append(f"COPY: Source '{src}' not found.")
            print(f"COPY: Source '{src}' not found.")
            return
        copied_value = copy.deepcopy(src_value)
        # Minimal Change: Set in 'heap' if 'heap' is in dest, else in 'data'
        if 'heap' in dest:
            self.memory['heap'][dest] = copied_value
        else:
            self.memory['data'][dest] = copied_value
        self.execution_log.append(f"COPY: Copied {src}({src_value}) to {dest}.")
        print(f"COPY: Copied {src}({src_value}) to {dest}.")

    def get_memory_value(self, key: str) -> Optional[Any]:
        """Retrieve a value from any memory segment."""
        for segment in ['data', 'heap']:
            if key in self.memory[segment]:
                return self.memory[segment][key]
        return None

    def set_memory_value(self, key: str, value: Any):
        """Set a value in the data segment."""
        self.memory['data'][key] = value

    # Data Structures and Arrays
    def instr_CREATE_ARRAY(self, data: List[Any], modifier: str):
        """Initialize an array in memory."""
        if len(data) != 2:
            self.execution_log.append("CREATE_ARRAY: Invalid data length.")
            print("CREATE_ARRAY: Invalid data length.")
            return
        array_name, size = data
        if not isinstance(size, int) or size <= 0:
            self.execution_log.append("CREATE_ARRAY: Size must be a positive integer.")
            print("CREATE_ARRAY: Size must be a positive integer.")
            return
        self.memory['heap'][array_name] = [0] * size
        self.execution_log.append(f"CREATE_ARRAY: Created array '{array_name}' with size {size}.")
        print(f"CREATE_ARRAY: Created array '{array_name}' with size {size}.")

    def instr_SET_INDEX(self, data: List[Any], modifier: str):
        """Assign a value to a specific index in an array."""
        if len(data) != 3:
            self.execution_log.append("SET_INDEX: Invalid data length.")
            print("SET_INDEX: Invalid data length.")
            return
        array_name, index, value = data
        if array_name not in self.memory['heap']:
            self.execution_log.append(f"SET_INDEX: Array '{array_name}' not found.")
            print(f"SET_INDEX: Array '{array_name}' not found.")
            return
        array = self.memory['heap'][array_name]
        if not isinstance(index, int) or not (0 <= index < len(array)):
            self.execution_log.append(f"SET_INDEX: Index {index} out of bounds for array '{array_name}'.")
            print(f"SET_INDEX: Index {index} out of bounds for array '{array_name}'.")
            return
        array[index] = value
        self.execution_log.append(f"SET_INDEX: Set '{array_name}[{index}]' to {value}.")
        print(f"SET_INDEX: Set '{array_name}[{index}]' to {value}.")

    def instr_GET_INDEX(self, data: List[Any], modifier: str):
        """Retrieve a value from a specific index in an array."""
        if len(data) != 3:
            self.execution_log.append("GET_INDEX: Invalid data length.")
            print("GET_INDEX: Invalid data length.")
            return
        dest, array_name, index = data
        if array_name not in self.memory['heap']:
            self.execution_log.append(f"GET_INDEX: Array '{array_name}' not found.")
            print(f"GET_INDEX: Array '{array_name}' not found.")
            return
        array = self.memory['heap'][array_name]
        if not isinstance(index, int) or not (0 <= index < len(array)):
            self.execution_log.append(f"GET_INDEX: Index {index} out of bounds for array '{array_name}'.")
            print(f"GET_INDEX: Index {index} out of bounds for array '{array_name}'.")
            return
        self.memory['data'][dest] = array[index]
        self.execution_log.append(f"GET_INDEX: Retrieved '{array_name}[{index}]' = {array[index]} into '{dest}'.")
        print(f"GET_INDEX: Retrieved '{array_name}[{index}]' = {array[index]} into '{dest}'.")

    # ------------------------------
    # Phase 2: Control Flow and Functionality
    # ------------------------------

    # Function Definitions and Calls
    def instr_DEFINE_FUNCTION(self, data: List[Any], modifier: str):
        """Define a reusable function with specified gene promoters."""
        if len(data) < 2:
            self.execution_log.append("DEFINE_FUNCTION: Invalid data length.")
            print("DEFINE_FUNCTION: Invalid data length.")
            return
        function_name = data[0].lower()
        gene_promoters = [promoter.lower() for promoter in data[1:]]
        # Register the function by associating its name with gene promoters
        self.memory['functions'][function_name] = gene_promoters
        self.execution_log.append(f"DEFINE_FUNCTION: Defined function '{function_name}' with promoters {gene_promoters}.")
        print(f"DEFINE_FUNCTION: Defined function '{function_name}' with promoters {gene_promoters}.")

    def instr_CALL_FUNCTION(self, data: List[Any], modifier: str):
        """Call a defined function by its name."""
        if len(data) != 1:
            self.execution_log.append("CALL_FUNCTION: Invalid data length.")
            print("CALL_FUNCTION: Invalid data length.")
            return
        function_name = data[0].lower()
        functions = self.memory.get('functions', {})
        if function_name not in functions:
            self.execution_log.append(f"CALL_FUNCTION: Function '{function_name}' not defined.")
            print(f"CALL_FUNCTION: Function '{function_name}' not defined.")
            return
        # Push current state to call stack
        self.memory['call_stack'].append(self.current_gene_promoter)
        self.execution_log.append(f"CALL_FUNCTION: Calling function '{function_name}'.")
        print(f"CALL_FUNCTION: Calling function '{function_name}'.")
        # Activate function genes
        for promoter in functions[function_name]:
            for gene in self.genome.functional_genes:
                if gene.promoter == promoter:
                    gene.active = True
                    self.current_gene_promoter = promoter
                    break

    def instr_RETURN(self, data: List[Any], modifier: str):
        """Return from the current function."""
        if len(data) != 0:
            self.execution_log.append("RETURN: Invalid data length.")
            print("RETURN: Invalid data length.")
            return
        if not self.memory['call_stack']:
            self.execution_log.append("RETURN: Call stack is empty.")
            print("RETURN: Call stack is empty.")
            return
        previous_gene_promoter = self.memory['call_stack'].pop()
        # Deactivate all genes except the one being returned to
        for gene in self.genome.functional_genes:
            gene.active = (gene.promoter == previous_gene_promoter)
        self.current_gene_promoter = previous_gene_promoter
        self.execution_log.append(f"RETURN: Returning to gene '{previous_gene_promoter}'.")
        print(f"RETURN: Returning to gene '{previous_gene_promoter}'.")

    # Exception Handling
    def instr_TRY(self, data: List[Any], modifier: str):
        """Begin a try block."""
        if len(data) != 0:
            self.execution_log.append("TRY: Invalid data length.")
            print("TRY: Invalid data length.")
            return
        self.memory['exception_stack'].append('TRY')
        self.execution_log.append("TRY: Beginning try block.")
        print("TRY: Beginning try block.")

    def instr_CATCH(self, data: List[Any], modifier: str):
        """Define a catch block."""
        if len(data) != 0:
            self.execution_log.append("CATCH: Invalid data length.")
            print("CATCH: Invalid data length.")
            return
        self.memory['exception_handlers'].append('CATCH')
        self.execution_log.append("CATCH: Beginning catch block.")
        print("CATCH: Beginning catch block.")

    def instr_THROW(self, data: List[Any], modifier: str):
        """Throw an exception."""
        if len(data) != 0:
            self.execution_log.append("THROW: Invalid data length.")
            print("THROW: Invalid data length.")
            return
        self.memory['exceptions'].append('EXCEPTION')
        self.execution_log.append("THROW: Exception thrown.")
        print("THROW: Exception thrown.")
        # Handle exception
        if 'TRY' in self.memory.get('exception_stack', []):
            self.memory['exception_stack'].remove('TRY')
            if self.memory['exception_handlers']:
                self.memory['exception_handlers'].pop()
                self.execution_log.append("Exception handled by CATCH block.")
                print("Exception handled by CATCH block.")
            else:
                self.execution_log.append("CATCH: No handler available.")
                print("CATCH: No handler available.")
        else:
            self.execution_log.append("Uncaught exception.")
            print("Uncaught exception.")

    # ------------------------------
    # Memory Regulation and Feedback
    # ------------------------------
    def regulate_genes(self, environment: Dict[str, Any]):
        """Activate or suppress genes based on environmental stimuli."""
        stimulus = environment.get("stimulus", "").lower()
        for gene in self.genome.functional_genes:
            if gene.promoter in stimulus:
                gene.active = True
            else:
                gene.active = False

    def evolve_interpreter(self):
        """Evolve the interpreter by mutating interpreter genes."""
        for gene in self.genome.interpreter_genes:
            gene.mutate_gene()

    def feedback_adjustment(self, feedback_signal: str):
        """Adjust the interpreter based on feedback."""
        if feedback_signal == "success":
            # Reinforce successful logic (e.g., increase activation of certain genes)
            self.execution_log.append("Feedback: Success. Reinforcing logic.")
            print("Feedback: Success. Reinforcing logic.")
            # Implement reinforcement strategies if needed
            # Example: Lower mutation rates to preserve successful configurations
            self.mutation_rates['codon'] = max(0.05, self.mutation_rates['codon'] * 0.95)
            self.mutation_rates['promoter'] = max(0.005, self.mutation_rates['promoter'] * 0.95)
            self.mutation_rates['terminator'] = max(0.005, self.mutation_rates['terminator'] * 0.95)
        elif feedback_signal == "failure":
            # Refine logic (e.g., mutate genes leading to failure)
            self.execution_log.append("Feedback: Failure. Refining logic.")
            print("Feedback: Failure. Refining logic.")
            self.genome.mutate_genome()
            # Implement adaptive mutation rates
            # Example: Increase mutation rates to explore more possibilities
            self.mutation_rates['codon'] = min(0.5, self.mutation_rates['codon'] * 1.05)
            self.mutation_rates['promoter'] = min(0.1, self.mutation_rates['promoter'] * 1.05)
            self.mutation_rates['terminator'] = min(0.1, self.mutation_rates['terminator'] * 1.05)

    # ------------------------------
    # Helper Methods
    # ------------------------------
    def execute_genome(self, environment: Dict[str, Any]):
        """Execute all active genes based on gene regulation."""
        self.regulate_genes(environment)
        for gene in self.genome.functional_genes:
            if gene.active:
                self.execute_gene(gene)

    def execute_gene(self, gene: Gene):
        """Execute a single gene."""
        for codon in gene.codons:
            if not self.running:
                break
            self.execute_codon(codon)


# ------------------------------
# Genetic Mechanisms
# ------------------------------

def mutate_gene(gene: Gene, mutation_rates: Dict[str, float] = None):
    """Mutate a gene."""
    gene.mutate_gene(mutation_rates=mutation_rates)

def evolve_interpreter(genome: Genome, mutation_rates: Dict[str, float] = None):
    """Mutate and refine the interpreter logic."""
    for gene in genome.interpreter_genes:
        mutate_gene(gene, mutation_rates=mutation_rates)

def crossover_genomes(genome1: Genome, genome2: Genome, crossover_rate: float = 0.3) -> Genome:
    """Perform crossover between two genomes."""
    return genome1.crossover(genome2, crossover_rate=crossover_rate)

def duplicate_gene(genome: Genome, gene_type: str):
    """Duplicate a gene within the genome."""
    genome.duplicate_gene(gene_type)

def regulate_genes(genome: Genome, environment: Dict[str, Any]):
    """Regulate gene activation based on environment."""
    for gene in genome.functional_genes:
        if gene.promoter in environment.get("stimulus", "").lower():
            gene.active = True
        else:
            gene.active = False


# ------------------------------
# Simulation Environment
# ------------------------------

class Simulation:
    def __init__(self, initial_genome: Genome, mutation_rate: float = 0.05):
        self.genome = initial_genome
        self.interpreter = GeneticInterpreter(self.genome)
        self.environment = {"stimulus": "start_reflex"}
        self.mutation_rate = mutation_rate  # Currently unused but can be integrated for dynamic mutation rates
        self.generation = 0
        self.lock = threading.Lock()

    def run(self):
        """Run the simulation indefinitely until manually stopped."""
        try:
            while True:
                with self.lock:
                    self.generation += 1
                    print(f"\n--- Generation {self.generation} ---")
                    self.interpreter.execute_genome(self.environment)
                    # Feedback mechanism based on specific criteria
                    success_conditions = [
                        self.interpreter.memory.get("x", 0) > 10,
                        self.interpreter.memory.get("y", 0) > 10,
                        self.interpreter.memory.get("z", 0) > 10
                    ]
                    if all(success_conditions):
                        feedback = "success"
                    else:
                        feedback = "failure"
                    self.interpreter.feedback_adjustment(feedback)
                    # Evolve interpreter
                    self.interpreter.evolve_interpreter()
                    # Mutate genome with adaptive mutation rates
                    self.genome.mutate_genome()
                    # Optionally, perform crossover or duplication
                    if self.generation % 3 == 0:
                        duplicate_gene(self.genome, "functional_genes")
                    # Update interpreter with the new genome
                    self.interpreter.genome = self.genome
                    # Optionally, add environment changes based on generation
                    self.update_environment(self.generation)
        except KeyboardInterrupt:
            print("\nSimulation stopped by user.")

    def update_environment(self, generation: int):
        """Update environmental stimuli based on generation or other criteria."""
        # For example, increase stimulus complexity over generations
        if generation == 3:
            self.environment["stimulus"] = "start_reflex start_new_gene"
            print("Environment Update: Added 'start_new_gene' to stimulus.")
        elif generation == 5:
            self.environment["stimulus"] = "start_reflex start_new_gene enhanced_stimulus"
            print("Environment Update: Added 'enhanced_stimulus' to stimulus.")
        elif generation == 7:
            self.environment["stimulus"] = "start_reflex start_new_gene enhanced_stimulus complex_stimulus"
            print("Environment Update: Added 'complex_stimulus' to stimulus.")
        elif generation == 10:
            self.environment["stimulus"] = "start_reflex start_new_gene enhanced_stimulus complex_stimulus ultra_stimulus"
            print("Environment Update: Added 'ultra_stimulus' to stimulus.")
        # Additional environment updates can be implemented here


# ------------------------------
# Testing and Verification
# ------------------------------

def test_stack_operations(interpreter: GeneticInterpreter):
    """Test stack operations: PUSH and POP."""
    # Push values onto the stack
    interpreter.execute_codon(Codon('PUSH', [100]))
    interpreter.execute_codon(Codon('PUSH', [200]))
    # Pop values from the stack into 'a' and 'b'
    interpreter.execute_codon(Codon('POP', ['a']))
    interpreter.execute_codon(Codon('POP', ['b']))
    # Verify
    assert interpreter.memory['data']['a'] == 200, "Test 1 Failed: 'a' should be 200."
    assert interpreter.memory['data']['b'] == 100, "Test 1 Failed: 'b' should be 100."
    print("Test 1 Passed: Stack Operations")

def test_logical_operations(interpreter: GeneticInterpreter):
    """Test logical operations: AND, OR, NOT, XOR."""
    # Initialize variables
    interpreter.execute_codon(Codon('LOAD', ['data', 'x', True]))
    interpreter.execute_codon(Codon('LOAD', ['data', 'y', False]))
    # Perform AND
    interpreter.execute_codon(Codon('AND', ['z', 'x', 'y']))
    assert interpreter.memory['data']['z'] == False, "Test 2 Failed: 'z' should be False."
    # Perform OR
    interpreter.execute_codon(Codon('OR', ['w', 'x', 'y']))
    assert interpreter.memory['data']['w'] == True, "Test 2 Failed: 'w' should be True."
    # Perform NOT
    interpreter.execute_codon(Codon('NOT', ['v', 'x']))
    assert interpreter.memory['data']['v'] == False, "Test 2 Failed: 'v' should be False."
    # Perform XOR
    interpreter.execute_codon(Codon('XOR', ['u', 'x', 'y']))
    assert interpreter.memory['data']['u'] == True, "Test 2 Failed: 'u' should be True."
    print("Test 2 Passed: Logical Operations")

def test_advanced_math_operations(interpreter: GeneticInterpreter):
    """Test advanced mathematical operations: EXP, SQRT, LOG."""
    # Initialize variables
    interpreter.execute_codon(Codon('LOAD', ['data', 'base', 2]))
    interpreter.execute_codon(Codon('LOAD', ['data', 'exponent', 3]))
    # Perform EXP
    interpreter.execute_codon(Codon('EXP', ['result', 'base', 'exponent']))
    assert interpreter.memory['data']['result'] == 8.0, "Test 3 Failed: 'result' should be 8.0."
    # Perform SQRT
    interpreter.execute_codon(Codon('SQRT', ['sqrt_result', 'result']))
    assert interpreter.memory['data']['sqrt_result'] == math.sqrt(8.0), "Test 3 Failed: 'sqrt_result' should be sqrt(8.0)."
    # Perform LOG
    interpreter.execute_codon(Codon('LOG', ['log_result', 'result']))
    assert math.isclose(interpreter.memory['data']['log_result'], math.log(8.0)), "Test 3 Failed: 'log_result' should be log(8.0)."
    print("Test 3 Passed: Advanced Mathematical Operations")

def test_data_movement_operations(interpreter: GeneticInterpreter):
    """Test data movement operations: MOVE and COPY."""
    # Initialize variables
    interpreter.execute_codon(Codon('LOAD', ['data', 'x', 50]))
    # Move x to y
    interpreter.execute_codon(Codon('MOVE', ['y', 'x']))
    assert interpreter.memory['data']['y'] == 50, "Test 4 Failed: 'y' should be 50."
    # Copy y to z
    interpreter.execute_codon(Codon('COPY', ['z', 'y']))
    assert interpreter.memory['data']['z'] == 50, "Test 4 Failed: 'z' should be 50."
    # Modify y and ensure z remains unchanged
    interpreter.execute_codon(Codon('LOAD', ['data', 'y', 100]))
    assert interpreter.memory['data']['z'] == 50, "Test 4 Failed: 'z' should remain 50 after modifying 'y'."
    print("Test 4 Passed: Data Movement Operations")

def test_data_structures_and_arrays(interpreter: GeneticInterpreter):
    """Test data structures and array operations: CREATE_ARRAY, SET_INDEX, GET_INDEX."""
    # Create an array 'arr' of size 5
    interpreter.execute_codon(Codon('CREATE_ARRAY', ['arr', 5]))
    assert interpreter.memory['heap']['arr'] == [0, 0, 0, 0, 0], "Test 5 Failed: 'arr' should be initialized to [0, 0, 0, 0, 0]."
    # Set index 2 to 100
    interpreter.execute_codon(Codon('SET_INDEX', ['arr', 2, 100]))
    assert interpreter.memory['heap']['arr'][2] == 100, "Test 5 Failed: 'arr[2]' should be 100."
    # Get index 2 into 'val'
    interpreter.execute_codon(Codon('GET_INDEX', ['val', 'arr', 2]))
    assert interpreter.memory['data']['val'] == 100, "Test 5 Failed: 'val' should be 100."
    # Attempt to get an out-of-bounds index
    interpreter.execute_codon(Codon('GET_INDEX', ['out', 'arr', 10]))
    assert 'out' not in interpreter.memory['data'], "Test 5 Failed: 'out' should not be set due to out-of-bounds."
    print("Test 5 Passed: Data Structures and Arrays")

def test_memory_segmentation(interpreter: GeneticInterpreter):
    """Test memory segmentation: data, stack, heap."""
    # Load data into 'data' segment
    interpreter.execute_codon(Codon('LOAD', ['data', 'a', 10]))
    assert interpreter.memory['data']['a'] == 10, "Test 6 Failed: 'a' should be 10."
    # Push to 'stack' segment
    interpreter.execute_codon(Codon('PUSH', [20]))
    assert interpreter.memory['stack'] == [20], "Test 6 Failed: 'stack' should contain [20]."
    # Move from 'data' to 'heap'
    interpreter.execute_codon(Codon('MOVE', ['heap_val', 'a']))
    assert interpreter.memory['heap']['heap_val'] == 10, "Test 6 Failed: 'heap_val' should be 10."
    # Copy from 'heap' to 'data'
    interpreter.execute_codon(Codon('COPY', ['b', 'heap_val']))
    assert interpreter.memory['data']['b'] == 10, "Test 6 Failed: 'b' should be 10."
    print("Test 6 Passed: Memory Segmentation")

def run_all_tests():
    """Run all defined tests to verify the interpreter's functionality."""
    genome = create_initial_genome()
    interpreter = GeneticInterpreter(genome)

    # Run Tests
    test_stack_operations(interpreter)
    test_logical_operations(interpreter)
    test_advanced_math_operations(interpreter)
    test_data_movement_operations(interpreter)
    test_data_structures_and_arrays(interpreter)
    test_memory_segmentation(interpreter)

    print("All Tests Passed Successfully!")


# ------------------------------
# Example Usage
# ------------------------------

def create_initial_genome() -> Genome:
    """
    Factory method to create a random, diverse Genome.
    
    This function generates a Genome instance with a randomized set of functional, control, 
    and interpreter genes. Each gene contains a random number of codons with varied instructions, 
    data parameters, and modifiers. The promoters and terminators are uniquely generated to 
    ensure validity and diversity. The function also incorporates random duplication of genes 
    to enhance genome complexity.
    
    Returns:
        Genome: A randomly generated Genome instance with diverse genetic configurations.
    """

    # ------------------------------
    # Helper Functions
    # ------------------------------
    
    def generate_unique_name(base: str, index: int) -> str:
        """
        Generate a unique name by appending a random suffix to the base string.
        
        Args:
            base (str): The base string for the name.
            index (int): The index number to ensure uniqueness.
        
        Returns:
            str: A unique name string.
        """
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
        return f"{base}_{index}_{suffix}"
    
    def generate_promoter(base: str, index: int) -> str:
        """
        Generate a unique promoter name based on a base and index.
        
        Args:
            base (str): The base string for the promoter.
            index (int): The index number to ensure uniqueness.
        
        Returns:
            str: A unique promoter name.
        """
        return generate_unique_name(base, index)
    
    def generate_terminator(promoter: str) -> str:
        """
        Generate a corresponding terminator name based on the promoter.
        
        Args:
            promoter (str): The promoter name to base the terminator on.
        
        Returns:
            str: A corresponding terminator name.
        """
        return f"end_{promoter}"
    
    def select_random_instruction() -> str:
        """
        Select a random valid instruction from the Codon VALID_INSTRUCTIONS.
        
        Returns:
            str: A randomly selected valid instruction.
        """
        return random.choice(Codon.VALID_INSTRUCTIONS)
    
    def select_random_modifier() -> str:
        """
        Select a random modifier from the predefined list.
        
        Returns:
            str: A randomly selected modifier.
        """
        modifiers = ["default", "conditional", "loop", "activate_if", "suppress_if"]
        return random.choice(modifiers)
    
    def generate_codon_data(instruction: str) -> List[Any]:
        """
        Generate appropriate data based on the instruction type.
        
        Args:
            instruction (str): The instruction name.
        
        Returns:
            List[Any]: A list of data elements corresponding to the instruction.
        """
        if instruction == 'LOAD':
            # LOAD: ['segment', 'key', value]
            segment = random.choice(['data', 'heap', 'stack'])
            key = ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 7)))
            value = random.randint(0, 100)
            return [segment, key, value]
        
        elif instruction in ['ADD', 'SUB', 'MULT', 'DIV', 'MOD']:
            # Arithmetic Operations: ['dest', 'op1', 'op2']
            dest = ''.join(random.choices(string.ascii_lowercase, k=3))
            op1 = ''.join(random.choices(string.ascii_lowercase, k=3))
            op2 = ''.join(random.choices(string.ascii_lowercase, k=3))
            return [dest, op1, op2]
        
        elif instruction in ['AND', 'OR', 'XOR']:
            # Logical Operations: ['dest', 'op1', 'op2']
            dest = ''.join(random.choices(string.ascii_lowercase, k=3))
            op1 = ''.join(random.choices(string.ascii_lowercase, k=3))
            op2 = ''.join(random.choices(string.ascii_lowercase, k=3))
            return [dest, op1, op2]
        
        elif instruction == 'NOT':
            # NOT Operation: ['dest', 'op']
            dest = ''.join(random.choices(string.ascii_lowercase, k=3))
            op = ''.join(random.choices(string.ascii_lowercase, k=3))
            return [dest, op]
        
        elif instruction in ['MOVE', 'COPY']:
            # Data Movement Operations: ['dest', 'src']
            dest = ''.join(random.choices(string.ascii_lowercase, k=3))
            src = ''.join(random.choices(string.ascii_lowercase, k=3))
            return [dest, src]
        
        elif instruction == 'EXP':
            # Exponentiation: ['dest', 'base', 'exponent']
            dest = ''.join(random.choices(string.ascii_lowercase, k=3))
            base = ''.join(random.choices(string.ascii_lowercase, k=3))
            exponent = ''.join(random.choices(string.ascii_lowercase, k=3))
            return [dest, base, exponent]
        
        elif instruction in ['SQRT', 'LOG']:
            # Advanced Mathematical Operations: ['dest', 'src']
            dest = ''.join(random.choices(string.ascii_lowercase, k=3))
            src = ''.join(random.choices(string.ascii_lowercase, k=3))
            return [dest, src]
        
        elif instruction == 'CREATE_ARRAY':
            # Array Creation: ['array_name', size]
            array_name = ''.join(random.choices(string.ascii_lowercase, k=5))
            size = random.randint(1, 20)
            return [array_name, size]
        
        elif instruction == 'SET_INDEX':
            # Array Index Assignment: ['array_name', index, value]
            array_name = ''.join(random.choices(string.ascii_lowercase, k=5))
            index = random.randint(0, 10)
            value = random.randint(0, 100)
            return [array_name, index, value]
        
        elif instruction == 'GET_INDEX':
            # Array Index Retrieval: ['dest', 'array_name', index]
            dest = ''.join(random.choices(string.ascii_lowercase, k=3))
            array_name = ''.join(random.choices(string.ascii_lowercase, k=5))
            index = random.randint(0, 10)
            return [dest, array_name, index]
        
        elif instruction == 'DEFINE_FUNCTION':
            # Function Definition: ['function_name', 'promoter1', ...]
            function_name = ''.join(random.choices(string.ascii_lowercase, k=5))
            num_promoters = random.randint(1, 3)
            promoters = [generate_promoter('start_func', random.randint(1, 100)) for _ in range(num_promoters)]
            return [function_name] + promoters
        
        elif instruction == 'CALL_FUNCTION':
            # Function Call: ['function_name']
            function_name = ''.join(random.choices(string.ascii_lowercase, k=5))
            return [function_name]
        
        elif instruction == 'RETURN':
            # Return from Function: []
            return []
        
        elif instruction in ['TRY', 'CATCH', 'THROW']:
            # Exception Handling Instructions: []
            return []
        
        elif instruction == 'NOP':
            # No Operation: []
            return []
        
        elif instruction == 'IF':
            # Conditional Jump: ['condition', 'target_promoter']
            var = random.choice(['x', 'y', 'z'])
            operator = random.choice(['>', '<', '==', '!=', '>=', '<='])
            value = random.randint(0, 100)
            condition = f"{var}{operator}{value}"
            target_promoter = ''.join(random.choices(string.ascii_lowercase, k=10))
            return [condition, target_promoter]
        
        else:
            # Default Case: []
            return []

    def generate_codons(num_codons: int) -> List[Codon]:
        """
        Generate a list of Codon instances with random instructions, data, and modifiers.
        
        Args:
            num_codons (int): The number of codons to generate.
        
        Returns:
            List[Codon]: A list of generated Codon instances.
        """
        codons = []
        for _ in range(num_codons):
            instruction = select_random_instruction()
            data = generate_codon_data(instruction)
            modifier = select_random_modifier()
            codon = Codon(instruction, data, modifier)
            codons.append(codon)
        return codons

    def generate_gene(gene_type: str, index: int) -> Gene:
        """
        Generate a Gene instance based on the gene type with random promoters, terminators, and codons.
        
        Args:
            gene_type (str): The type of gene ('functional', 'control', 'interpreter').
            index (int): The index number to ensure unique promoter and terminator.
        
        Returns:
            Gene: A generated Gene instance.
        """
        promoter_bases = {
            'functional': ['start_func', 'init', 'execute', 'process', 'run'],
            'control': ['control', 'manage', 'handle'],
            'interpreter': ['interpreter', 'decode', 'translate']
        }
        
        base_promoter = random.choice(promoter_bases[gene_type])
        promoter = generate_promoter(base_promoter, index)
        terminator = generate_terminator(promoter)
        
        # Define the number of codons based on gene type
        if gene_type == 'functional':
            num_codons = random.randint(3, 10)
        elif gene_type == 'control':
            num_codons = random.randint(1, 5)
        elif gene_type == 'interpreter':
            num_codons = random.randint(1, 5)
        else:
            num_codons = random.randint(1, 5)
        
        codons = generate_codons(num_codons)
        return Gene(promoter=promoter, codons=codons, terminator=terminator)

    def duplicate_existing_genes(genes: List[Gene], num_duplicates: int, gene_type: str):
        """
        Duplicate existing genes to introduce redundancy and complexity.
        
        Args:
            genes (List[Gene]): The list of genes to duplicate from.
            num_duplicates (int): The number of duplicates to create.
            gene_type (str): The type of gene ('functional', 'control', 'interpreter') for logging.
        """
        for _ in range(num_duplicates):
            if not genes:
                break
            gene_to_duplicate = random.choice(genes)
            duplicated_gene = copy.deepcopy(gene_to_duplicate)
            # Modify promoter and terminator to maintain uniqueness
            duplicated_gene.promoter = ''.join(random.choices(string.ascii_lowercase, k=10))
            duplicated_gene.terminator = f"end_{duplicated_gene.promoter}"
            genes.append(duplicated_gene)
            print(f"Duplication: Duplicated a '{gene_type}' gene with new promoter '{duplicated_gene.promoter}'.")

    # ------------------------------
    # Genome Generation
    # ------------------------------
    
    # Define the number of genes for each category
    num_functional_genes = random.randint(5, 15)
    num_control_genes = random.randint(2, 5)
    num_interpreter_genes = random.randint(1, 3)
    
    # Generate functional genes
    functional_genes = []
    for i in range(1, num_functional_genes + 1):
        gene = generate_gene('functional', i)
        functional_genes.append(gene)
    
    # Generate control genes
    control_genes = []
    for i in range(1, num_control_genes + 1):
        gene = generate_gene('control', i)
        control_genes.append(gene)
    
    # Generate interpreter genes
    interpreter_genes = []
    for i in range(1, num_interpreter_genes + 1):
        gene = generate_gene('interpreter', i)
        interpreter_genes.append(gene)
    
    # Introduce gene duplication for complexity
    duplication_chance = 0.3  # 30% chance to duplicate genes
    for gene_list, gene_type in zip(
        [functional_genes, control_genes, interpreter_genes],
        ['functional', 'control', 'interpreter']
    ):
        if random.random() < duplication_chance:
            num_duplicates = random.randint(1, 3)
            duplicate_existing_genes(gene_list, num_duplicates, gene_type)
    
    # Assemble the genome
    genome = Genome(
        functional_genes=functional_genes,
        control_genes=control_genes,
        interpreter_genes=interpreter_genes
    )
    
    return genome


def main():
    """Main function to initialize genome, run tests, and start the simulation."""
    # Create initial genome
    initial_genome = create_initial_genome()
    
    # Initialize interpreter
    interpreter = GeneticInterpreter(initial_genome)
    
    # Run all tests
    run_all_tests()
    
    # Initialize simulation
    simulation = Simulation(initial_genome)
    simulation_thread = threading.Thread(target=simulation.run, daemon=True)
    simulation_thread.start()
    print("Simulation started. Press Ctrl+C to stop.")
    
    # Keep the main thread alive to allow simulation to run
    try:
        while simulation_thread.is_alive():
            simulation_thread.join(timeout=1)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")

if __name__ == "__main__":
    main()
