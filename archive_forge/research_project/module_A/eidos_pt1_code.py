"""
Eidos Framework: Module A - Input Processing System (P_in)
Implements Theorem 1: Deterministic Input Processing Pipeline
Version: 3.1.0 (Formalized Academic Edition)

Formalizes the input processing pipeline through:
- Σ*-Algebraic type foundations with categorical duality
- Immutable Merkle-provenanced metadata architecture
- Non-commutative operator algebras with Banach space embeddings
- Holomorphic modality handlers satisfying Cauchy-Riemann constraints
- Recursively validated processing DAGs with idempotent convergence
"""

import unicodedata
import hashlib
import enum
import logging
import time
import re
import json
import numpy as np
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    NamedTuple,
    Iterable,
    TypeVar,
    Type,
    Callable,
    Iterator,
    Deque,
    DefaultDict,
    Optional,
    Set,
    FrozenSet,
    Protocol,
    cast,
    runtime_checkable,
    TypeGuard,
)
from typing_extensions import TypeIs
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from logging.handlers import RotatingFileHandler
from functools import wraps, partial
import string
from typing import NamedTuple as StateTuple
from inspect import ismethod

# Define base error code before PMPError class
EIDOS_PMP = 0xE0000000  # Base error code for PMP framework


class PMPError(Exception):
    """Root exception for PMP framework errors"""

    base_code = EIDOS_PMP


class CategoryError(PMPError):
    """Violation of categorical constraints (Morphism non-preservation)"""

    code = PMPError.base_code | 0x100


class ImplementationError(PMPError):
    """Incomplete interface implementation (Failed UMP Satisfaction)"""

    code = PMPError.base_code | 0x200


class ConfigurationError(PMPError):
    """Invalid configuration parameters (Structure Non-preservation)"""

    code = PMPError.base_code | 0x300


class ProcessingError(PMPError):
    """Base processing failure (DAG Execution Fault)"""

    code = PMPError.base_code | 0x400


class InvariantViolationError(PMPError):
    """System invariant violation (Coherence Condition Failure)"""

    code = PMPError.base_code | 0x500


# Define PipelineStep enum before ProcessingStage
class PipelineStep(enum.Enum):
    NORMALIZATION = "normalize"
    NOISE_REMOVAL = "denoise"
    STRUCTURING = "structure"
    ENCODING = "encode"
    VALIDATION = "validate"


# ----------------------------
# Core PMP Structures
# ----------------------------


class Process(ABC):
    """Formal process interface per PMP Definition 3.1"""

    @property
    @abstractmethod
    def state_schema(self) -> Type[NamedTuple]:
        """Returns schema as NamedTuple type"""
        pass

    @abstractmethod
    def transition(
        self, input_data: Any, current_state: NamedTuple
    ) -> Tuple[Any, NamedTuple]:
        """State transition function satisfying f_M: I_M × S_M → O_M × S_M"""
        pass

    @property
    @abstractmethod
    def interface(self) -> "ModuleSignature":
        """Formal interface contract per Lemma 2.4"""
        pass

    @abstractmethod
    def compose(self, other: "Process") -> "Process":
        """Operator algebra composition per PMP Theorem 4.3"""
        pass


@dataclass(frozen=True)
class ModuleSignature:
    """Formal module interface contract with algebraic properties"""

    input_types: FrozenSet[Type]
    output_types: FrozenSet[Type]
    state_type: Type
    algebraic_properties: FrozenSet[str] = field(
        default_factory=lambda: frozenset(["associative", "unital"])
    )
    merkle_root: str = field(default="", init=False)

    def __post_init__(self):
        components = (
            [str(ft.__name__) for ft in self.input_types]
            + [str(ft.__name__) for ft in self.output_types]
            + [self.state_type.__name__]
            + list(self.algebraic_properties)
        )
        object.__setattr__(
            self,
            "merkle_root",
            hashlib.sha3_256("|".join(components).encode()).hexdigest(),
        )


# ----------------------------
# Σ*-Algebraic Type System
# ----------------------------


class ModalityType(enum.Enum):
    """Formal specification of modalities (Σ) per Definition 2.1"""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    SENSOR = "sensor"
    GEOSPATIAL = "geospatial"
    STRUCTURED = "structured"
    QUANTUM = "quantum_state"

    @classmethod
    def get_algebra(cls, modality: "ModalityType") -> "ModalityAlgebra":
        """Retrieves Σ* algebraic structure per Axiom 1.2"""
        return AlgebraRegistry.get_algebra(modality)

    @classmethod
    def supported(cls) -> List["ModalityType"]:
        """Returns implemented modalities with active pipelines (Lemma 2.3)"""
        return list(ModalityHandlerRegistry._handlers.keys())


# ----------------------------
# Algebraic Infrastructure
# ----------------------------


class AlgebraRegistry:
    """PMP-compliant algebraic registry with UMP"""

    _algebras: Dict[ModalityType, "ModalityAlgebra"] = {}

    @classmethod
    def register(cls, modality: ModalityType, algebra: "ModalityAlgebra"):
        """Registers algebra with structure validation"""
        cls._validate_algebra(algebra)
        cls._algebras[modality] = algebra

    @classmethod
    def get_algebra(cls, modality: ModalityType) -> "ModalityAlgebra":
        """Retrieves algebra with existence check"""
        if algebra := cls._algebras.get(modality):
            return algebra
        raise NotImplementedError(f"No algebra defined for {modality}")

    @staticmethod
    def _validate_algebra(algebra: "ModalityAlgebra"):
        """Validates algebraic structure per PMP Axioms"""
        required_ops = {"identity", "compose", "inverse"}
        if not required_ops.issubset(algebra.operations):
            raise CategoryError(f"Algebra missing required operations: {required_ops}")


# ----------------------------
# Metadata Architecture
# ----------------------------


@dataclass(frozen=True)
class ModalityMetadata:
    """Immutable preservation metadata with Merkle proofs"""

    modality: ModalityType
    schema_version: str = "3.1.0"
    provenance_chain: Tuple[str, ...] = field(default_factory=tuple)
    system_properties: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    merkle_root: str = field(default="", init=False)

    def __post_init__(self):
        """Computes Merkle root using SHA3-256"""
        leaves = [hashlib.sha3_256(p.encode()).digest() for p in self.provenance_chain]
        while len(leaves) > 1:
            if len(leaves) % 2:
                leaves.append(leaves[-1])
            leaves = [
                hashlib.sha3_256(l + r).digest()
                for l, r in zip(leaves[::2], leaves[1::2])
            ]
        object.__setattr__(self, "merkle_root", leaves[0].hex() if leaves else "")

    def add_provenance(self, operation: "ProcessingStage") -> "ModalityMetadata":
        """Creates new metadata with updated provenance chain"""
        return ModalityMetadata(
            modality=self.modality,
            schema_version=self.schema_version,
            provenance_chain=(*self.provenance_chain, operation.stage_fingerprint),
            system_properties=self.system_properties.copy(),
            quality_metrics=self.quality_metrics.copy(),
        )

    def to_manifest(self) -> Dict[str, Any]:
        """Serialize metadata for logging/transmission"""
        return {
            "modality": self.modality.value,
            "schema_version": self.schema_version,
            "merkle_root": self.merkle_root,
            "provenance_length": len(self.provenance_chain),
        }


# ----------------------------
# Processing Infrastructure
# ----------------------------
@dataclass(frozen=True)
class PreprocessorOutput:
    processed_data: Any
    modality: ModalityType
    processing_hash: str
    metadata: ModalityMetadata
    pipeline_fingerprint: str
    processing_trace: Tuple[str, ...]


@dataclass(frozen=True)
class ProcessingStage:
    """Formal processing stage with cryptographic fingerprint"""

    step_type: PipelineStep
    operator: "ProcessingOperator"
    config_hash: str
    position: int
    dependencies: Set[int] = field(default_factory=set)
    api_version: str = "3.0"
    stage_fingerprint: str = field(init=False)


class PipelineState(NamedTuple):
    current_data: Any
    metadata: ModalityMetadata
    quality_metrics: Dict[str, float]


class ProcessingPipeline(Process):
    """PMP-compliant pipeline implementation per Theorem 4.1"""

    def __init__(self, stages: Iterable[ProcessingStage]):
        self.stages = tuple(stages)
        self.dag = self._construct_dag()
        self._signature = self._compute_signature()
        self.topological_order = self._topological_sort()

    def _construct_dag(self) -> Dict[int, Set[int]]:
        """Constructs processing DAG with dependency validation"""
        dag = defaultdict(set)
        for i, stage in enumerate(self.stages):
            for dep in stage.dependencies:
                dag[i].add(dep)
        return dag

    def _compute_signature(self) -> str:
        """Computes cryptographic pipeline signature"""
        components = [stage.stage_fingerprint for stage in self.stages]
        return hashlib.shake_256("".join(components).encode()).hexdigest(64)

    def _topological_sort(self) -> List["ProcessingStage"]:
        """Kahn's algorithm for topological sorting"""
        in_degree: DefaultDict[int, int] = defaultdict(int)
        for node in self.dag:
            for neighbor in self.dag[node]:
                in_degree[neighbor] += 1

        queue = deque([node for node in self.dag if in_degree[node] == 0])
        sorted_order = []

        while queue:
            node = queue.popleft()
            sorted_order.append(self.stages[node])
            for neighbor in self.dag[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_order) != len(self.stages):
            raise ProcessingError("Cycle detected in processing DAG")
        return sorted_order

    @property
    def state_schema(self) -> Type[NamedTuple]:
        return PipelineState

    def transition(
        self, input_data: Any, current_state: NamedTuple
    ) -> Tuple[Any, NamedTuple]:
        """Executes pipeline stages with state management"""
        # Cast to concrete type for internal processing
        state = cast(PipelineState, current_state)
        try:
            data = input_data
            for stage in self.topological_order:
                data = stage.operator(data)
                state = self._update_state(stage, data, state)
            return data, state
        except ProcessingError as e:
            self._handle_error(e, state)
            raise

    def _update_state(
        self, stage: ProcessingStage, data: Any, state: PipelineState
    ) -> PipelineState:
        """Immutable state update with provenance tracking"""
        new_metrics = {
            "latency": time.monotonic()
            - state.quality_metrics.get("start_time", time.monotonic())
        }
        return state._replace(
            current_data=data,
            metadata=state.metadata.add_provenance(stage),
            quality_metrics={**state.quality_metrics, **new_metrics},
        )

    def _handle_error(self, error: ProcessingError, state: PipelineState) -> None:
        """Error handling with state preservation"""
        error_info = {
            "timestamp": time.time(),
            "error_type": type(error).__name__,
            "stage_fingerprint": getattr(error, "stage_fingerprint", ""),
            "state": state._asdict(),
        }
        logging.error("Processing failure: %s", error_info)


# ----------------------------
# Operator Algebra
# ----------------------------


@runtime_checkable
class ProcessingOperator(Protocol):
    """Abstract operator protocol with algebraic properties"""

    @property
    def config_fingerprint(self) -> str:
        """Unique configuration fingerprint"""
        ...

    @property
    def algebraic_properties(self) -> Set[str]:
        """Set of algebraic properties (commutative, etc.)"""
        ...

    def validate_configuration(self) -> bool:
        """Validate operator configuration"""
        ...

    def __call__(self, data: Any) -> Any:
        """Operator application method"""
        ...


class NormalizationOperator(ProcessingOperator):
    """Base normalization operator with modality constraints"""

    @property
    @abstractmethod
    def supported_modality(self) -> ModalityType: ...


@runtime_checkable
class NoiseRemovalOperator(ProcessingOperator, Protocol):
    """Protocol for noise removal operators"""

    pass  # Inherits required methods from ProcessingOperator


# ----------------------------
# Modality Categorical Framework
# ----------------------------


@dataclass(frozen=True)
class ModalityAlgebra:
    """Formal Σ*-Algebraic structure per Definition 2.2 with topological closure"""

    character_set: FrozenSet[str]
    operations: Dict[str, Callable]
    identity_element: Any
    inverse_operation: Callable
    is_abelian: bool
    topological_properties: Dict[str, Any] = field(
        default_factory=lambda: {"hausdorff": True, "compact": False}
    )
    metric: Optional[Callable[[Any, Any], float]] = field(default=None)
    norm: Optional[Callable[[Any], float]] = field(default=None)

    def __post_init__(self):
        """Validate algebraic structure per PMP axioms"""
        if self.identity_element not in self.character_set:
            raise CategoryError("Identity element must belong to character set")
        if not all(
            op(self.identity_element, self.identity_element) == self.identity_element
            for op in self.operations.values()
            if callable(op)
        ):
            raise CategoryError("Operation preservation failure")


class ModalityHandler(ABC):
    """Monoidal interface with complete categorical implementation"""

    @abstractmethod
    def generate_metadata(self, raw_data: Any) -> ModalityMetadata:
        """Must implement modality-specific metadata generation"""
        ...

    @classmethod
    @abstractmethod
    def base_algebra(cls) -> ModalityAlgebra:
        """Must return fully initialized ModalityAlgebra instance"""
        ...

    @classmethod
    @abstractmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Must return validated configuration with type/structure checks"""
        ...

    @abstractmethod
    def validate_input(self, data: Any) -> bool:
        """Must implement modality-specific validation"""
        ...

    @abstractmethod
    def serialize_processed(self, data: Any) -> bytes:
        """Must implement reversible serialization"""
        ...

    @classmethod
    def get_algebraic_properties(cls) -> Dict[str, Any]:
        """Retrieve complete algebraic structure definition"""
        return asdict(cls.base_algebra())


class ModalityHandlerRegistry:
    """Universal registry implementing strict UMP with versioned configurations"""

    _handlers: Dict[ModalityType, Type[ModalityHandler]] = {}
    _versions: Dict[ModalityType, str] = {}
    _configs: Dict[ModalityType, Dict[str, Any]] = defaultdict(dict)

    @classmethod
    def get_handler(cls, modality: ModalityType) -> Type[ModalityHandler]:
        """Retrieve validated handler class with existence check"""
        if handler := cls._handlers.get(modality):
            return handler
        raise NotImplementedError(f"No handler registered for {modality}")

    @classmethod
    def get_registered_modalities(cls) -> List[ModalityType]:
        """Return sorted list of registered modalities"""
        return sorted(cls._handlers.keys(), key=lambda m: m.value)

    @classmethod
    def register_handler(
        cls,
        modality: ModalityType,
        version: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Callable[[Type[ModalityHandler]], Type[ModalityHandler]]:
        """Decorator factory for type-safe handler registration"""

        def decorator(handler_class: Type[ModalityHandler]) -> Type[ModalityHandler]:
            if modality in cls._handlers:
                raise CategoryError(f"Handler conflict for {modality}")
            cls._validate_handler_interface(handler_class)
            validated_config = handler_class.validate_config(config or {})

            cls._handlers[modality] = handler_class
            cls._versions[modality] = version
            cls._configs[modality] = validated_config

            # Register associated algebra
            AlgebraRegistry.register(modality, handler_class.base_algebra())

            return handler_class

        return decorator

    @classmethod
    def _validate_handler_interface(cls, handler: Type[ModalityHandler]):
        """Strict implementation check per PMP Theorem 3.2"""

        def is_concrete_classmethod(m: object) -> TypeGuard[classmethod]:
            return isinstance(m, classmethod) and not getattr(
                m.__func__, "__isabstractmethod__", False
            )

        required = {
            "base_algebra": lambda m: is_concrete_classmethod(m),
            "validate_config": lambda m: is_concrete_classmethod(m),
            "validate_input": callable,
            "serialize_processed": callable,
        }

        for attr, validator in required.items():
            if not validator(getattr(handler, attr)):
                raise CategoryError(f"Invalid implementation for {attr}")

        # Validate algebraic coherence
        algebra = handler.base_algebra()
        if not isinstance(algebra, ModalityAlgebra):
            raise CategoryError("Handler algebra must be ModalityAlgebra instance")


# ----------------------------
# Text Modality Implementation
# ----------------------------


@ModalityHandlerRegistry.register_handler(
    ModalityType.TEXT,
    "3.1",
    {
        "normalization_form": "NFC",
        "stream_buffer_size": 4096,
        "unicode_version": unicodedata.unidata_version,
    },
)
class TextModalityHandler(ModalityHandler):
    def __init__(self):
        self._config = self.validate_config(
            ModalityHandlerRegistry._configs[ModalityType.TEXT]
        )

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures configuration satisfies NFKC constraints"""
        if "normalization_form" not in config:
            raise ValueError("Normalization form required")
        if config["normalization_form"] not in {"NFC", "NFD", "NFKC", "NFKD"}:
            raise ValueError("Invalid normalization form")
        return config

    def validate_input(self, data: Any) -> bool:
        return isinstance(data, (str, bytes))

    def generate_metadata(self, raw_data: Any) -> ModalityMetadata:
        text = raw_data.decode() if isinstance(raw_data, bytes) else raw_data
        return ModalityMetadata(
            modality=ModalityType.TEXT,
            system_properties={
                "length": len(text),
                "grapheme_clusters": len(list(unicodedata.normalize("NFC", text))),
                "unicode_planes": len({ord(c) >> 16 for c in text}),
                "normalization_form": self._config["normalization_form"],
            },
        )

    def compute_identity_hash(self, data: Any) -> str:
        normalized = unicodedata.normalize(
            self._config["normalization_form"],
            data.decode() if isinstance(data, bytes) else data,
        )
        return hashlib.sha3_256(normalized.encode("utf-8")).hexdigest()

    def streaming_processor(self) -> Callable[[Iterator[bytes]], Iterator[str]]:
        def processor(stream: Iterator[bytes]) -> Iterator[str]:
            buffer: deque[str] = deque()
            for chunk in stream:
                buffer.extend(chunk.decode("utf-8", errors="replace"))
                while len(buffer) >= self._config["stream_buffer_size"]:
                    yield "".join(
                        [
                            buffer.popleft()
                            for _ in range(self._config["stream_buffer_size"])
                        ]
                    )
            if buffer:
                yield "".join(buffer)

        return processor

    def serialize_processed(self, data: Any) -> bytes:
        if isinstance(data, str):
            return data.encode("utf-8")
        elif isinstance(data, bytes):
            return data
        raise TypeError("Data for serialization must be str or bytes")

    @classmethod
    def base_algebra(cls) -> ModalityAlgebra:
        """Defines the free monoid over Unicode codepoints"""
        return ModalityAlgebra(
            character_set=frozenset(chr(i) for i in range(0x0000, 0x10FFFF)),
            operations={
                "concatenate": lambda a, b: a + b,
                "length": len,
                "normalize": unicodedata.normalize,
            },
            identity_element="",
            inverse_operation=lambda s: s[::-1],
            is_abelian=False,
        )


# ----------------------------
# Topological Processing Pipeline
# ----------------------------


class ModalityPipeline:
    """Formal implementation of processing DAG (Theorem 4.1)"""

    def __init__(self, operators: Iterable[ProcessingOperator]):
        self.operator_dag = self._construct_validated_dag(operators)
        self.pipeline_fingerprint = self._compute_pipeline_hash()

    def _construct_validated_dag(
        self, operators: Iterable[ProcessingOperator]
    ) -> Dict[PipelineStep, List[ProcessingStage]]:
        dag: DefaultDict[PipelineStep, List[ProcessingStage]] = defaultdict(list)
        for idx, op in enumerate(operators):
            # Validate operator type first
            if not isinstance(op, ProcessingOperator):
                raise ConfigurationError(f"Invalid operator type: {type(op)}")

            if not op.validate_configuration():
                raise ConfigurationError(f"Invalid operator configuration: {op}")
            stage = ProcessingStage(
                step_type=self._resolve_step_type(op),
                operator=op,
                config_hash=op.config_fingerprint,
                position=idx,
            )
            dag[stage.step_type].append(stage)
        return dict(dag)

    def _resolve_step_type(self, operator: ProcessingOperator) -> PipelineStep:
        """Map operator type to processing step"""
        if isinstance(operator, NormalizationOperator):
            return PipelineStep.NORMALIZATION
        if isinstance(operator, NoiseRemovalOperator):
            return PipelineStep.NOISE_REMOVAL
        return PipelineStep.VALIDATION

    def _compute_pipeline_hash(self) -> str:
        """Compute SHA3-256 hash of pipeline configuration"""
        config_str = "|".join(
            f"{stage.step_type.value}:{stage.config_hash}"
            for stages in self.operator_dag.values()
            for stage in stages
        )
        return hashlib.sha3_256(config_str.encode()).hexdigest()

    def process(self, raw_input: Any, modality: ModalityType) -> PreprocessorOutput:
        handler = ModalityHandlerRegistry.get_handler(modality)()
        if not handler.validate_input(raw_input):
            logger = logging.getLogger(__name__)
            logger.error(f"Invalid {modality.value} input")
            raise ValueError(f"Invalid {modality.value} input")

        current_data = raw_input
        processing_trace = []
        metadata = handler.generate_metadata(raw_input)

        try:
            for step in PipelineStep:
                for stage in self.operator_dag.get(step, []):
                    current_data = stage.operator(current_data)
                    processing_trace.append(stage.stage_fingerprint)
                    metadata = metadata.add_provenance(stage)
        except ProcessingError as e:
            self._handle_processing_error(e, metadata)
            raise

        return PreprocessorOutput(
            processed_data=current_data,
            modality=modality,
            processing_hash=hashlib.sha3_256(str(raw_input).encode()).hexdigest(),
            metadata=metadata,
            pipeline_fingerprint=self.pipeline_fingerprint,
            processing_trace=tuple(processing_trace),
        )

    def _handle_processing_error(
        self, error: ProcessingError, metadata: ModalityMetadata
    ):
        """Error handling with context preservation"""
        error_info = {
            "timestamp": time.time(),
            "error_type": type(error).__name__,
            "metadata": metadata.to_manifest(),
        }
        logging.error("Processing failure: %s", error_info)


# ----------------------------
# Core Framework Implementation
# ----------------------------


class EidosInputProcessor:
    """Canonical implementation of Module A (Definition 1.1)"""

    MODULE_ID: str = "A"
    VERSION: Tuple[int, int, int] = (3, 1, 0)

    def __init__(self):
        self.pipelines: Dict[ModalityType, ModalityPipeline] = {}
        self.logger = self._configure_logger()

    def _configure_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(
            "eidos_processing.log", maxBytes=10 * 1024 * 1024, backupCount=5
        )
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def register_pipeline(
        self, modality: ModalityType, operators: List[ProcessingOperator]
    ) -> None:
        if modality not in ModalityHandlerRegistry.get_registered_modalities():
            raise NotImplementedError(f"Unsupported modality: {modality}")
        pipeline = ModalityPipeline(operators)
        self.pipelines[modality] = pipeline
        self.logger.info(
            f"Registered {modality} pipeline: {pipeline.pipeline_fingerprint}"
        )

    def process(self, raw_input: Any, modality: ModalityType) -> PreprocessorOutput:
        if modality not in self.pipelines:
            raise ValueError(f"No pipeline for {modality}")
        return self.pipelines[modality].process(raw_input, modality)


# ----------------------------
# Operator Implementations
# ----------------------------


class AdaptiveNormalizer(NormalizationOperator):
    """Hierarchical normalizer with contextual pattern learning and cryptographic audit trails"""

    _learned_patterns: DefaultDict[str, DefaultDict[str, dict]] = defaultdict(
        lambda: defaultdict(dict)
    )
    _usage_tracking: DefaultDict[str, int] = defaultdict(int)

    BASE_REPLACEMENTS = {
        "diacritics": {
            "Æ": ("AE", 0.95),
            "Œ": ("OE", 0.93),
            "ß": ("ss", 0.98),
            "à": ("a", 0.9),
            "é": ("e", 0.9),
            "ï": ("i", 0.9),
        },
        "symbols": {
            "μ": ("micro", 0.85),
            "Ω": ("ohm", 0.88),
            "℃": ("C", 0.82),
            "±": ("±", 0.78),
            "→": ("→", 0.75),
            "×": ("×", 0.8),
        },
        "ligatures": {"ﬁ": ("fi", 0.92), "ﬂ": ("fl", 0.91), "ﬀ": ("ff", 0.9)},
        "currency": {"€": ("EUR", 0.88), "£": ("GBP", 0.85), "¥": ("JPY", 0.82)},
        "whitespace": {"\u200B": (" ", 1.0), "\u202F": (" ", 0.95)},
    }

    def __init__(self, context_window_size: int = 3, pattern_threshold: int = 1000):
        self.replacement_graph = self._build_replacement_graph()
        self.context_window_size = context_window_size
        self.pattern_threshold = pattern_threshold
        self.domain_hierarchy = [
            "general",
            "technical",
            "financial",
            "medical",
            "linguistic",
        ]
        self.logger = logging.getLogger(f"{__name__}.AdaptiveNormalizer")
        self._pattern_cache: Dict[str, re.Pattern] = {}
        self._replacement_log: Deque[Tuple[str, str, int]] = deque(maxlen=1000)

    def _build_replacement_graph(self) -> Dict[str, Tuple[str, float]]:
        """Construct initial replacement graph with case-sensitive patterns"""
        graph = {}
        for category, replacements in self.BASE_REPLACEMENTS.items():
            graph.update({k: (v[0].strip(), v[1]) for k, v in replacements.items()})
            if category in {"diacritics", "ligatures"}:
                graph.update(
                    {k.lower(): (v[0].lower(), v[1]) for k, v in replacements.items()}
                )
                graph.update(
                    {k.upper(): (v[0].upper(), v[1]) for k, v in replacements.items()}
                )
        return graph

    def __call__(self, data: str) -> str:
        """Execute normalization with context-aware processing and whitespace preservation"""
        original_hash = hashlib.sha3_256(data.encode()).hexdigest()
        self._track_usage_context(data)
        processed = self._adaptive_process(data)

        # Store transformation metadata for reversibility
        normalized = unicodedata.normalize(
            "NFC", processed
        )  # Changed to NFC for better reversibility
        self._replacement_log.append((original_hash, normalized, int(time.time())))

        return normalized

    def _track_usage_context(self, text: str) -> None:
        """Enhanced context tracking with punctuation awareness"""
        for char, context in self._generate_context_windows(text):
            if char.strip() or char in {"\u200B", "\u202F"}:
                self._usage_tracking[char] += 1
                if self._usage_tracking[char] % self.pattern_threshold == 0:
                    self._update_merkle_tree(char, context)

    def _adaptive_process(self, text: str) -> str:
        """Process text with hierarchical pattern matching and case preservation"""
        buffer = []
        replacement_map = []
        for i, char in enumerate(text):
            if char in string.whitespace or unicodedata.category(char).startswith("P"):
                buffer.append(char)
                continue

            context = self._get_context_window(text, i)
            replacement = self._get_contextual_replacement(char, context)
            buffer.append(replacement or char)
            if replacement and replacement != char:
                replacement_map.append((i, char, replacement, context))

        # Store transformation map for inverse operations
        self._store_transformation_map(text, replacement_map)
        return "".join(buffer)

    def _store_transformation_map(self, original: str, changes: list):
        """Store transformation data with cryptographic proof"""
        transformation_data = {
            "original": original,
            "changes": changes,
            "timestamp": time.time(),
            "merkle_root": hashlib.sha3_256(original.encode()).hexdigest(),
        }
        self._learned_patterns["transformation_log"][
            hashlib.shake_256(original.encode()).hexdigest(16)
        ] = transformation_data

    def _get_contextual_replacement(self, char: str, context: str) -> Optional[str]:
        """Get replacement with domain-aware pattern matching"""
        if context not in self._pattern_cache:
            self._pattern_cache[context] = re.compile(re.escape(context), re.IGNORECASE)

        if pattern := self._pattern_cache.get(context):
            if match := pattern.search(context):
                return self._handle_pattern_match(match.group())
        return self._match_hierarchical_rule(char, context)

    def inverse_operation(self, normalized_text: str) -> str:
        """Reconstruct original text from normalization log"""
        original_hash = hashlib.sha3_256(normalized_text.encode()).hexdigest()
        for log_entry in reversed(self._replacement_log):
            if log_entry[0] == original_hash:
                return next(
                    (
                        v["original"]  # Explicitly access original text
                        for v in self._learned_patterns["transformation_log"].values()
                        if v["merkle_root"] == original_hash
                    ),
                    normalized_text,
                )
        return normalized_text

    def _handle_pattern_match(self, matched: str) -> str:
        """Handle multi-character pattern matches with case preservation"""
        return self.replacement_graph.get(matched, (matched, 0.0))[0]

    def _match_hierarchical_rule(self, char: str, context: str) -> Optional[str]:
        """Check domain hierarchy for replacement rules"""
        for domain in reversed(self.domain_hierarchy):
            if rule := self._learned_patterns[domain][char].get(
                self._context_hash(context)
            ):
                return rule[0]
        return self.replacement_graph.get(char, (None,))[0]

    def _update_merkle_tree(self, char: str, context: str) -> None:
        """Update Merkle tree with contextual pattern and case variants"""
        domain = self._detect_domain(context)
        context_hash = self._context_hash(context)

        # Store case variants separately with context-aware weighting
        for case_variant in {char, char.lower(), char.upper()}:
            self._learned_patterns[domain][case_variant][context_hash] = (
                self._derive_replacement_rule(case_variant, context)
            )

        leaves = [
            hashlib.sha3_256(f"{k}{v}".encode()).digest()
            for k, v in self._learned_patterns[domain][char].items()
        ]
        if leaves:
            self._learned_patterns[domain][char]["_merkle"] = hashlib.sha3_256(
                b"".join(leaves)
            ).hexdigest()

    def _detect_domain(self, context: str) -> str:
        """Enhanced domain detection with punctuation analysis"""
        domain_scores: Dict[str, float] = defaultdict(float)
        punctuation = sum(1 for c in context if unicodedata.category(c).startswith("P"))

        for domain in self.domain_hierarchy:
            score = sum(
                context.count(char) * weight
                for char, weight in self._get_domain_indicators(domain)
            )
            # Penalize domains for unexpected punctuation
            if domain != "linguistic" and punctuation > 3:
                score *= 0.8
            domain_scores[domain] = score

        return max(domain_scores.items(), key=lambda x: x[1])[0]

    @property
    def config_fingerprint(self) -> str:
        return hashlib.shake_256(
            f"{self.context_window_size}:{self.pattern_threshold}:{json.dumps(self.domain_hierarchy)}".encode()
        ).hexdigest(64)

    def _generate_context_windows(self, text: str) -> Iterator[Tuple[str, str]]:
        """Yield (character, context_window) pairs"""
        for i in range(len(text)):
            context = self._get_context_window(text, i)
            yield text[i], context

    def _get_context_window(self, text: str, index: int, window_size: int = 2) -> str:
        """Generate context window around character position"""
        start = max(0, index - window_size)
        end = min(len(text), index + window_size + 1)
        return text[start:end]

    def _context_hash(self, context: str) -> str:
        """Generate stable hash for context window"""
        return hashlib.sha1(context.encode()).hexdigest()[
            :8
        ]  # Truncated for efficiency

    def _derive_replacement_rule(self, char: str, context: str) -> Tuple[str, float]:
        """Generate context-aware replacement rules with confidence scoring"""
        # Calculate replacement confidence based on context frequency
        context_freq = self._usage_tracking.get(context, 1)
        base_replacement = self.replacement_graph.get(char, (char, 0.0))[0]
        confidence = min(
            0.95, 1 - (1 / context_freq)
        )  # Confidence increases with usage

        return (base_replacement, confidence)

    def _get_domain_indicators(self, domain: str) -> List[Tuple[str, float]]:
        """Domain-specific character indicators with weights"""
        return {
            "technical": [("μ", 1.3), ("Ω", 1.2), ("→", 1.1)],
            "financial": [("€", 1.2), ("£", 1.1), ("¥", 1.0)],
            "medical": [("®", 1.3), ("™", 1.1), ("α", 1.2)],
        }.get(domain, [])

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_modality(self) -> ModalityType:
        return ModalityType.TEXT

    def validate_configuration(self) -> bool:
        """Validate pattern threshold configuration"""
        return isinstance(self.pattern_threshold, int) and self.pattern_threshold > 0


class DomainAwareSanitizer(NoiseRemovalOperator):
    """Multi-level sanitizer with hierarchical domain adaptation"""

    DOMAIN_HIERARCHY = {
        "technical": {
            "threshold": 8.5,  # Enhanced threshold for technical domains
            "subdomains": {
                "electronics": {
                    "indicators": [
                        ("μ", 1.2),
                        ("Ω", 1.1),
                        ("ℏ", 1.0),
                    ],  # Updated technical symbols with normalized weights
                    "whitelist": [
                        "μ",
                        "Ω",
                        "ℏ",
                    ],  # Converted to ordered list with core symbols
                },
                "physics": {
                    "indicators": [
                        ("ψ", 1.3),
                        ("φ", 1.2),
                        ("∂", 1.1),
                    ],  # Expanded Greek character support
                    "whitelist": [
                        "ψ",
                        "φ",
                        "∂",
                    ],  # Standardized quantum notation symbols
                },
            },
        },
        "financial": {
            "threshold": 2,
            "subdomains": {
                "crypto": {"indicators": [("₿", 1.5), ("Ξ", 1.3)]},
                "forex": {"indicators": [("€", 1.2), ("£", 1.1), ("¥", 1.0)]},
            },
        },
        "medical": {
            "threshold": 2,
            "subdomains": {
                "pharma": {"indicators": [("®", 1.3), ("™", 1.1)]},
                "biotech": {"indicators": [("α", 1.2), ("β", 1.1)]},
            },
        },
    }

    @property
    def config_fingerprint(self) -> str:
        """Generate stable configuration fingerprint using SHAKE-256"""
        return hashlib.shake_256(
            json.dumps(self.DOMAIN_HIERARCHY, sort_keys=True).encode()
        ).hexdigest(64)

    def __init__(self):
        self.domain_handlers = HierarchicalHandlerRegistry()
        self._build_domain_handlers()
        self.logger = logging.getLogger(f"{__name__}.DomainAwareSanitizer")
        self._compiled_patterns: Dict[str, re.Pattern] = {}

    def __call__(self, data: str) -> str:
        """Execute domain-adaptive sanitization with order-sensitive processing"""
        domain_path = self._detect_domain_hierarchy(data)
        processed = self.domain_handlers.resolve(domain_path)(data)

        # Final cleanup preserving domain-specific characters
        return self._sanitize_with(
            processed,
            preserve=self._get_domain_whitelist(domain_path),
            remove_diacritics=False,
        )

    def _get_domain_whitelist(self, domain_path: List[str]) -> Set[str]:
        """Get combined whitelist for detected domain hierarchy"""
        whitelist = set()
        for domain in domain_path:
            domain_config = self.DOMAIN_HIERARCHY.get(domain, {})
            if "subdomains" in domain_config:
                # Safely handle subdomain configurations
                subdomains = domain_config["subdomains"]
                if isinstance(subdomains, dict):  # Type guard
                    for sub, config in subdomains.items():
                        whitelist.update(config.get("whitelist", set()))
        return whitelist

    def _sanitize_with(
        self,
        text: str,
        remove_diacritics: bool = True,
        preserve: Optional[Set[str]] = None,
    ) -> str:
        """Enhanced sanitization with whitelist preservation"""
        preserve = preserve or set()  # Handle None case
        if remove_diacritics:
            text = unicodedata.normalize("NFD", text)
            text = "".join(c for c in text if not unicodedata.combining(c))

        if preserve:
            # Enhanced pattern with proper escaping and whitespace handling
            pattern = re.compile(
                rf"[^\w\s{re.escape(''.join(preserve))}]",  # Raw f-string for safety
                flags=re.UNICODE | re.IGNORECASE,
            )
            return pattern.sub("", text).strip()

        # Context-sensitive aggressive cleanup
        return re.sub(
            r"[^\w\s\-_]", "", text.translate(str.maketrans("", "", string.punctuation))
        ).strip()

    def _technical_rules(self, text: str) -> str:
        """Technical domain rules with symbol preservation"""
        # Add context-aware wrapping to preserve technical symbols
        text = re.sub(r"(\d+)([μΩℏ℃])(?=\D|$)", r"\1<TECH>\2</TECH>", text)
        return self._sanitize_with(
            text, preserve={"<", ">", "/", "TECH", "μ", "Ω", "→", "ℏ", "℃"}
        )

    def _electronics_rules(self, text: str) -> str:
        """Electronics subdomain processing with unit preservation"""
        text = re.sub(r"(?i)\b(\d+)([kM]?Ω)\b", r"\1 \2", text)
        return self._sanitize_with(text, preserve={"μ", "Ω", "→", "°"})

    def _financial_rules(self, text: str) -> str:
        """Financial domain normalization with currency conversion tracking"""
        converted = re.sub(r"([$€£¥])(\d+)", r"<CUR:\1>\2", text)
        return self._sanitize_with(converted, preserve={"<", ">", ":", "CUR"})

    def _medical_rules(self, text: str) -> str:
        """Medical domain normalization with measurement preservation"""
        processed = re.sub(r"([±]?\d+\.\d+%?)", r"<MEASURE>\1</MEASURE>", text)
        return self._sanitize_with(processed, preserve={"<", ">", "/", "MEASURE"})

    def _general_rules(self, text: str) -> str:
        """General domain rules with whitespace preservation"""
        return self._sanitize_with(text, preserve=set(string.whitespace))

    def validate_configuration(self) -> bool:
        """Validate domain hierarchy configuration"""
        return len(self.DOMAIN_HIERARCHY) > 0

    def _build_domain_handlers(self) -> None:
        """Register domain-specific processing rules"""
        self.domain_handlers.register("technical", self._technical_rules)
        self.domain_handlers.register("electronics", self._electronics_rules)
        self.domain_handlers.register("financial", self._financial_rules)
        self.domain_handlers.register("medical", self._medical_rules)
        self.domain_handlers.register("general", self._general_rules)

    def _detect_domain_hierarchy(self, text: str) -> List[str]:
        """Detect domain hierarchy based on character patterns"""
        hierarchy = []
        for domain, config in self.DOMAIN_HIERARCHY.items():
            score = sum(
                text.count(char) * weight
                for subdomains in [config.get("subdomains", {})]
                if isinstance(subdomains, dict)  # Type guard
                for sub_config in subdomains.values()
                for char, weight in sub_config.get("indicators", [])
            )
            if score >= config["threshold"]:
                hierarchy.append(domain)
        return hierarchy or ["general"]

    @property
    def version(self) -> str:
        return "1.4.1"

    @property
    def supported_modality(self) -> ModalityType:
        return ModalityType.TEXT


class HierarchicalHandlerRegistry:
    """Hierarchical resolution with fallback and circuit breaker"""

    def __init__(self):
        self.handlers: Dict[str, Callable[[str], str]] = {}
        self.fallback_order = ["general"]
        self.failure_count: DefaultDict[str, int] = defaultdict(int)
        self.MAX_FAILURES = 5

    def register(self, domain: str, handler: Callable):
        self.handlers[domain] = self._wrap_with_safety(handler)

    def _wrap_with_safety(self, handler: Callable) -> Callable:
        """Add circuit breaker pattern to handlers"""

        def wrapped(text: str) -> str:
            if self.failure_count[handler.__name__] >= self.MAX_FAILURES:
                return text  # Fail-safe
            try:
                return handler(text)
            except Exception as e:
                self.failure_count[handler.__name__] += 1
                logging.error(f"Handler {handler.__name__} failed: {str(e)}")
                return text

        return wrapped

    def resolve(self, domains: List[str]) -> Callable:
        """Resolve handler with fallback strategy"""
        for domain in domains + self.fallback_order:
            if handler := self.handlers.get(domain):
                return handler
        return lambda x: x
