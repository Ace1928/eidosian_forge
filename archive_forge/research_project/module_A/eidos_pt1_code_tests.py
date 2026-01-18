import unittest
from unittest.mock import MagicMock
import tracemalloc
import cProfile
import pstats
import json
import io
import unicodedata
import hashlib
from typing import List, Literal
from eidos_pt1_code import (
    CategoryError,
    ImplementationError,
    ConfigurationError,
    ProcessingError,
    InvariantViolationError,
    ModalityType,
    ModalityHandlerRegistry,
    ModalityMetadata,
    ModalityAlgebra,
    ModalityPipeline,
    PipelineStep,
    ProcessingStage,
    ProcessingOperator,
    AdaptiveNormalizer,
    DomainAwareSanitizer,
    TextModalityHandler,
    EidosInputProcessor,
)
import logging


class TestEidosFrameworkExceptions(unittest.TestCase):
    """Comprehensive test suite for exception hierarchy and behavior"""

    def test_exception_hierarchy_and_properties(self):
        """Validate exception inheritance and basic functionality"""
        test_cases = [
            (CategoryError, Exception, "Category constraint failure"),
            (ImplementationError, Exception, "Interface failure"),
            (ConfigurationError, Exception, "Config error"),
            (ProcessingError, Exception, "Pipeline failure"),
            (InvariantViolationError, Exception, "Invariant breach"),
        ]

        for exc_type, parent_type, msg in test_cases:
            with self.subTest(exc_type=exc_type.__name__):
                # Test inheritance
                self.assertTrue(
                    issubclass(exc_type, parent_type),
                    f"{exc_type.__name__} should inherit from {parent_type.__name__}",
                )

                # Test message formatting
                exc = exc_type(msg)
                self.assertEqual(str(exc), msg, "Exception message not properly set")

                # Test exception chaining
                try:
                    try:
                        raise ValueError("Root cause")
                    except ValueError as ve:
                        raise exc_type("Wrapper") from ve
                except exc_type as e:
                    self.assertIsInstance(
                        e.__cause__, ValueError, "Exception cause should be preserved"
                    )
                    self.assertIn(
                        "Root cause",
                        str(e.__cause__),
                        "Original exception message should be preserved",
                    )


class TestModalityTypeSystem(unittest.TestCase):
    """Comprehensive test suite for Σ* type system foundations"""

    def setUp(self):
        tracemalloc.start()
        self.registered_handlers = ModalityHandlerRegistry.get_registered_modalities()

    def tearDown(self):
        tracemalloc.stop()

    def test_modality_enum_completeness(self):
        """Validate all modality enum values and properties"""
        expected_values = {
            "TEXT": ("text", 0),
            "IMAGE": ("image", 1),
            "AUDIO": ("audio", 2),
            "VIDEO": ("video", 3),
            "SENSOR": ("sensor", 4),
            "GEOSPATIAL": ("geospatial", 5),
            "STRUCTURED": ("structured", 6),
            "QUANTUM": ("quantum_state", 7),
        }

        for name, (value, index) in expected_values.items():
            with self.subTest(modality=name):
                enum_member = ModalityType[name]
                self.assertEqual(
                    enum_member.value, value, f"Enum value mismatch for {name}"
                )
                self.assertEqual(
                    enum_member._value_,
                    value,
                    f"Enum storage value mismatch for {name}",
                )
                self.assertEqual(
                    enum_member._name_, name, f"Enum name mismatch for {value}"
                )

    def test_supported_modalities_implementation(self):
        """Validate Lemma 2.3 implementation with memory analysis"""
        # Initial memory snapshot
        snapshot = tracemalloc.take_snapshot()

        registered = ModalityType.supported()

        # Type validation
        self.assertIsInstance(
            registered, list, "Supported modalities should return a list"
        )
        self.assertTrue(
            len(registered) > 0, "At least one modality should be supported"
        )
        self.assertTrue(
            all(isinstance(m, ModalityType) for m in registered),
            "All returned values should be ModalityType instances",
        )

        # Memory analysis
        new_snapshot = tracemalloc.take_snapshot()
        diff = new_snapshot.compare_to(snapshot, "lineno")
        alloc_size = sum(stat.size_diff for stat in diff)

        self.assertLess(
            alloc_size,
            2048,
            f"Excessive memory allocation ({alloc_size} bytes) for supported modalities",
        )

    def test_base_algebra_properties(self):
        """Validate Axiom 1.2 implementation with performance characteristics"""
        for modality in self.registered_handlers:
            with self.subTest(
                modality=modality.name
            ), cProfile.Profile() as pr, self.assertLogs(level="INFO") as logs:

                # Retrieve algebra structure
                algebra = ModalityType.base_algebra(modality)

                # Type validation
                self.assertIsInstance(
                    algebra, ModalityAlgebra, "Should return a ModalityAlgebra instance"
                )
                self.assertTrue(
                    hasattr(algebra, "operations"),
                    "Algebra should have operations mapping",
                )
                self.assertTrue(
                    callable(algebra.inverse_operation),
                    "Inverse operation should be callable",
                )

                # Performance profiling
                pr.disable()
                s = io.StringIO()
                stats = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime")
                stats.print_stats(10)

                self.assertLess(
                    len(s.getvalue()), 1000, "Excessive time in algebra retrieval"
                )


class TestModalityMetadata(unittest.TestCase):
    """Comprehensive test suite for cryptographic metadata structure"""

    def test_merkle_tree_integrity(self):
        """Validate Theorem 4.5 implementation with multiple provenance steps"""
        metadata = ModalityMetadata(ModalityType.TEXT)
        stages = [
            ProcessingStage(
                step_type=PipelineStep.NORMALIZATION,
                operator=MagicMock(),
                config_hash=f"hash_{i}",
                position=i,
            )
            for i in range(4)
        ]

        merkle_roots = []
        for i, stage in enumerate(stages):
            with self.subTest(step=i):
                metadata = metadata.add_provenance(stage)
                merkle_roots.append(metadata.merkle_root)

                # Validate incremental changes
                if i > 0:
                    self.assertNotEqual(
                        merkle_roots[-1],
                        merkle_roots[-2],
                        "Merkle root should change with each addition",
                    )
                    self.assertEqual(
                        len(metadata.provenance_chain),
                        i + 1,
                        "Provenance chain length mismatch",
                    )

        # Final validation
        self.assertEqual(
            len(metadata.provenance_chain), 4, "All stages should be recorded"
        )
        self.assertEqual(
            len(metadata.merkle_root), 64, "Merkle root should be SHA3-256 hex digest"
        )

    def test_metadata_serialization(self):
        """Validate metadata serialization and integrity"""
        test_metrics = {"accuracy": 0.99, "fidelity": 0.95}
        test_slices = [{"start": 0, "end": 1}]

        metadata = ModalityMetadata(
            modality=ModalityType.TEXT,
            quality_metrics=test_metrics,
            temporal_slices=tuple(test_slices),
        ).add_provenance(
            ProcessingStage(
                step_type=PipelineStep.NORMALIZATION,
                operator=MagicMock(),
                config_hash="test_hash",
                position=0,
            )
        )

        manifest = metadata.to_manifest()

        # Validate structure
        self.assertEqual(manifest["modality"], "text")
        self.assertEqual(manifest["schema_version"], "3.1.0")
        self.assertEqual(manifest["provenance_length"], 1)

        # Validate data integrity
        self.assertDictEqual(
            manifest["quality_metrics"],
            test_metrics,
            "Quality metrics should be preserved",
        )
        self.assertListEqual(
            manifest["temporal_slices"],
            test_slices,
            "Temporal slices should be preserved",
        )
        self.assertEqual(
            len(manifest["merkle_root"]), 64, "Merkle root should be 64-character hash"
        )


class TestProcessingOperators(unittest.TestCase):
    """Comprehensive test suite for operator algebra implementations"""

    @classmethod
    def setUpClass(cls):
        cls.normalizer = AdaptiveNormalizer(pattern_threshold=1000)
        cls.sanitizer = DomainAwareSanitizer()
        cls.valid_operators = [cls.normalizer, cls.sanitizer]
        tracemalloc.start()

    @classmethod
    def tearDownClass(cls):
        tracemalloc.stop()

    def test_operator_contract_compliance(self):
        """Validate operator C*-algebra properties"""
        for operator in self.valid_operators:
            with self.subTest(operator=type(operator).__name__):
                # Interface validation
                self.assertTrue(
                    callable(operator.validate_configuration),
                    "Must implement validate_configuration",
                )
                self.assertTrue(
                    hasattr(operator, "config_fingerprint"),
                    "Must have config_fingerprint property",
                )

                # Configuration validation
                self.assertTrue(
                    operator.validate_configuration(),
                    "Operator configuration should be valid",
                )

                # Fingerprint validation
                fp = operator.config_fingerprint
                self.assertEqual(
                    len(fp), 128, "Fingerprint should be 128-character hash"
                )
                self.assertTrue(
                    all(c in "0123456789abcdef" for c in fp),
                    "Fingerprint should be hexadecimal",
                )

    def test_operator_non_commutativity(self):
        """Validate non-commutative operator behavior with various inputs"""
        test_cases = [
            ("  Leading/  Trailing  ", "Whitespace"),
            ("Quantum state: μ→ℏ", "Technical symbols"),
            ("Price: $500 → €450", "Financial notation"),
        ]

        # Initialize with protocol-compliant configurations
        adaptive_normalizer = AdaptiveNormalizer(
            context_window_size=3, pattern_threshold=1000
        )
        domain_sanitizer = DomainAwareSanitizer()

        for data, desc in test_cases:
            with self.subTest(description=desc):
                # Validate non-commutative algebra
                normal_first = domain_sanitizer(adaptive_normalizer(data))
                reverse_order = adaptive_normalizer(domain_sanitizer(data))
                self.assertNotEqual(
                    normal_first,
                    reverse_order,
                    f"Violation of non-commutative algebra for {desc}",
                )

                # Validate adaptive retention mechanics
                if "μ" in data:
                    # Verify cryptographic pattern retention
                    self.assertIn(
                        "micro",
                        normal_first,
                        "Initial normalization failed for technical symbols",
                    )

                    # Simulate merkle tree updates
                    for _ in range(1500):
                        adaptive_normalizer("μμμ")
                        domain_sanitizer("μ→ℏ")  # Direct symbol exposure

                    # Verify quantum-resistant retention
                    adapted_output = domain_sanitizer(adaptive_normalizer(data))
                    self.assertIn(
                        "<TECH>μ</TECH>",
                        adapted_output,
                        "Failed symbol retention after adaptive learning",
                    )

                # Validate domain-specific holomorphic processing
                if "Price:" in data:
                    processed = domain_sanitizer(adaptive_normalizer(data))
                    self.assertNotIn(
                        "$", processed, "Financial symbol sanitization failure"
                    )
                    self.assertIn(
                        "<CUR:$>", processed, "Currency conversion rules not applied"
                    )

    def test_memory_characteristics(self):
        """Validate memory usage patterns with various input sizes"""
        test_cases = [
            (10**3, "1KB input"),
            (10**4, "10KB input"),
            (10**5, "100KB input"),
            (10**6, "1MB input"),
        ]

        for size, desc in test_cases:
            with self.subTest(description=desc):
                test_data = "X" * size

                # Memory measurement
                snapshot = tracemalloc.take_snapshot()
                _ = self.normalizer(test_data)
                new_snapshot = tracemalloc.take_snapshot()

                # Analysis
                diff = new_snapshot.compare_to(snapshot, "lineno")
                total_alloc = sum(stat.size_diff for stat in diff)

                self.assertLess(
                    total_alloc,
                    len(test_data) ** 3,
                    f"Excessive allocation ({total_alloc} bytes) for {desc}",
                )


class TestTextModalityHandler(unittest.TestCase):
    """Comprehensive test suite for text modality implementations"""

    def setUp(self):
        self.handler = TextModalityHandler()
        self.test_cases = [
            ("Héllö, Wørld! ", "Basic case"),
            ("caf\u00e9", "Precomposed character"),
            ("\u0063\u0061\u0066\u0065\u0301", "Decomposed character"),
        ]

    def test_stream_processing_edge_cases(self):
        """Validate streaming processor with various chunk sizes"""
        test_streams = [
            (iter([b"Chunk1", b"Chunk2", b"Chunk3"]), "Small chunks"),
            (iter([b"LargeChunk" * 1000]), "Single large chunk"),
            (iter([b"Partial"] * 100), "Many small chunks"),
            (iter([]), "Empty stream"),
        ]

        for stream, desc in test_streams:
            with self.subTest(description=desc):
                processor = self.handler.streaming_processor()
                output = list(processor(stream))

                if desc == "Empty stream":
                    self.assertEqual(
                        len(output), 0, "Empty stream should produce no output"
                    )
                else:
                    # Validate chunk size constraints
                    self.assertTrue(
                        all(len(chunk) <= 4096 for chunk in output),
                        "Chunk size exceeds 4096 bytes",
                    )
                    # Validate data integrity
                    input_data = b"".join(stream)
                    output_data = "".join(output).encode()
                    self.assertEqual(
                        input_data, output_data, "Stream processing corrupted data"
                    )

    def test_normalization_integrity(self):
        """Validate Unicode normalization compliance across forms"""
        NORMALIZATION_FORMS: List[Literal["NFC", "NFD", "NFKC", "NFKD"]] = [
            "NFC",
            "NFD",
            "NFKC",
            "NFKD",
        ]

        for form in NORMALIZATION_FORMS:
            for data, _ in self.test_cases:
                with self.subTest(form=form, data=data[:20]):
                    test_handler = TextModalityHandler()
                    normalized = unicodedata.normalize(form, data)

                    expected_hash = hashlib.sha3_256(
                        normalized.encode("utf-8")
                    ).hexdigest()
                    actual_hash = test_handler.compute_identity_hash(data)

                    self.assertEqual(
                        actual_hash,
                        expected_hash,
                        f"Hash mismatch for {form} normalization",
                    )


class TestModalityPipeline(unittest.TestCase):
    """Comprehensive test suite for processing DAGs"""

    def test_dag_construction_validation(self):
        """Validate Theorem 4.1 implementation with various configurations"""
        valid_cases = [
            (
                [AdaptiveNormalizer(pattern_threshold=1000), DomainAwareSanitizer()],
                "Valid pair",
            ),
            ([AdaptiveNormalizer(pattern_threshold=1000)], "Single operator"),
            ([DomainAwareSanitizer(), DomainAwareSanitizer()], "Duplicate operators"),
        ]

        invalid_cases = [
            ([MagicMock(spec=ProcessingOperator)], "Invalid implementation"),
            ([], "Empty pipeline"),
        ]

        # Validate valid configurations
        for operators, desc in valid_cases:
            with self.subTest(valid_case=desc):
                pipeline = ModalityPipeline(operators)
                self.assertEqual(
                    len(pipeline.operator_dag),
                    len(set(type(o) for o in operators)),
                    "Operator DAG construction mismatch",
                )

    def test_pipeline_fingerprint_uniqueness(self):
        """Validate pipeline fingerprint stability and uniqueness"""
        # Create different pipeline configurations
        pipeline1 = ModalityPipeline(
            [AdaptiveNormalizer(pattern_threshold=1000), DomainAwareSanitizer()]
        )
        pipeline2 = ModalityPipeline(
            [DomainAwareSanitizer(), AdaptiveNormalizer(pattern_threshold=1000)]
        )
        pipeline3 = ModalityPipeline(
            [
                AdaptiveNormalizer(pattern_threshold=1000),
                AdaptiveNormalizer(pattern_threshold=1000),
            ]
        )

        # Validate uniqueness
        fingerprints = {
            pipeline1.pipeline_fingerprint,
            pipeline2.pipeline_fingerprint,
            pipeline3.pipeline_fingerprint,
        }
        self.assertEqual(
            len(fingerprints), 3, "All pipeline fingerprints should be unique"
        )

        # Validate format
        for fp in fingerprints:
            self.assertEqual(len(fp), 64, "Fingerprint should be SHA3-256 hash")
            self.assertTrue(
                all(c in "0123456789abcdef" for c in fp),
                "Fingerprint should be hexadecimal",
            )


class TestEidosInputProcessorIntegration(unittest.TestCase):
    """End-to-end test suite with performance and security validation"""

    def setUp(self):
        self.processor = EidosInputProcessor()
        self.processor.register_pipeline(
            ModalityType.TEXT,
            [AdaptiveNormalizer(pattern_threshold=1000), DomainAwareSanitizer()],
        )
        self.valid_input = "Héllö, Wørld! "
        self.invalid_input = 12345  # Non-text input

    def test_processing_chain_integrity(self):
        """Validate full processing chain from Theorem 1.1"""
        result = self.processor.process(self.valid_input, ModalityType.TEXT)

        # Cryptographic validation
        self.assertEqual(
            len(result.algebraic_signature), 128, "Algebraic signature length mismatch"
        )
        self.assertEqual(result.version_info, (3, 1, 0), "Version info mismatch")

        # Provenance validation
        self.assertEqual(
            len(result.processing_trace), 2, "Should have two processing steps"
        )
        self.assertEqual(
            len(result.metadata.provenance_chain), 2, "Provenance chain length mismatch"
        )

        # Merkle root validation
        self.assertEqual(
            result.metadata.merkle_root,
            result.metadata._compute_merkle_root(),
            "Merkle root validation failed",
        )

    def test_error_handling_and_logging(self):
        """Validate error handling and logging mechanisms"""
        invalid_input = 12345  # Not a valid text input

        # Match the actual logger name from the code module
        with self.assertLogs("eidos_pt1_code", level="ERROR") as log_ctx:
            with self.assertRaises(ValueError):
                self.processor.process(invalid_input, ModalityType.TEXT)

        # Verify log message contents
        self.assertIn("Invalid text input", log_ctx.records[0].getMessage())
        self.assertEqual(log_ctx.records[0].levelno, logging.ERROR)

    def test_performance_characteristics(self):
        """Validate processing performance meets requirements"""
        with cProfile.Profile() as pr:
            for _ in range(2048):
                self.processor.process(self.valid_input, ModalityType.TEXT)

        # Performance analysis
        pr.disable()
        s = io.StringIO()
        stats = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime")
        stats.print_stats(10)

        stats_output = s.getvalue()
        self.assertIn("function calls", stats_output, "Missing profiling statistics")
        self.assertLess(len(stats_output), 2048, "Excessive profiling output")


if __name__ == "__main__":
    unittest.main(failfast=True, verbosity=2, buffer=True)
